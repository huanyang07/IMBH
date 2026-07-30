"""Positive fixed-band energy-transfer diagnostics.

This module is audit-only.  It measures total and invariant-subspace energy
stored in a fixed physical cell band.  It does not alter the causal-inner
residual, tangent, or time integrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_one_way_scattering import (
    causal_integrate_frozen_window,
)


@dataclass(frozen=True)
class CausalPositiveBandEnergyHistory:
    """Positive total and family-resolved energy in one cell band."""

    total_energy: np.ndarray
    family_energy: np.ndarray
    maximum_family_partition_relative_defect: float
    minimum_total_energy: float
    minimum_family_energy: float


@dataclass(frozen=True)
class CausalNormalizedArrivalEnergy:
    """Time-averaged and peak arrival energy normalized by initial energy."""

    total_time_average: np.ndarray
    family_time_average: np.ndarray
    peak_total: np.ndarray
    maximum_integrated_partition_relative_defect: float


def causal_positive_band_energy_history(
    physical_history: np.ndarray,
    *,
    log_edges: np.ndarray,
    energy_metrics: np.ndarray,
    projectors: np.ndarray,
    lower_face: int,
    upper_face: int,
) -> CausalPositiveBandEnergyHistory:
    """Evaluate positive total and projector-resolved energy in a cell band."""

    history = np.asarray(physical_history, dtype=float)
    edges = np.asarray(log_edges, dtype=float)
    energy = np.asarray(energy_metrics, dtype=float)
    families = np.asarray(projectors, dtype=float)
    lower = int(lower_face)
    upper = int(upper_face)
    if (
        history.ndim != 4
        or history.shape[-1] < 1
        or edges.shape != (history.shape[2] + 1,)
        or energy.shape
        != (history.shape[2], history.shape[3], history.shape[3])
        or families.shape
        != (
            history.shape[2],
            history.shape[3],
            history.shape[3],
            history.shape[3],
        )
        or not 0 <= lower < upper <= history.shape[2]
        or np.any(~np.isfinite(history))
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
        or np.any(~np.isfinite(energy))
        or np.any(~np.isfinite(families))
    ):
        raise ValueError("positive band-energy inputs are invalid")

    region = slice(lower, upper)
    widths = np.diff(edges)[region]
    states = history[:, :, region]
    total = 0.5 * np.einsum(
        "tcni,nij,tcnj,n->tc",
        states,
        energy[region],
        states,
        widths,
        optimize=True,
    )
    projected_metrics = np.einsum(
        "nfki,nkl,nflj->nfij",
        families[region],
        energy[region],
        families[region],
        optimize=True,
    )
    family = 0.5 * np.einsum(
        "tcni,nfij,tcnj,n->tcf",
        states,
        projected_metrics,
        states,
        widths,
        optimize=True,
    )
    partition_scale = max(
        float(np.max(np.abs(total))),
        float(np.max(np.abs(family))),
        np.finfo(float).tiny,
    )
    partition = float(
        np.max(np.abs(np.sum(family, axis=-1) - total))
        / partition_scale
    )
    positivity_scale = max(
        float(np.max(np.abs(total))),
        float(np.max(np.abs(family))),
        np.finfo(float).tiny,
    )
    tolerance = 1.0e-12 * positivity_scale
    minimum_total = float(np.min(total))
    minimum_family = float(np.min(family))
    if minimum_total < -tolerance or minimum_family < -tolerance:
        raise ValueError("band energy is not positive to roundoff")
    total = np.maximum(total, 0.0)
    family = np.maximum(family, 0.0)
    return CausalPositiveBandEnergyHistory(
        total_energy=np.asarray(total),
        family_energy=np.asarray(family),
        maximum_family_partition_relative_defect=partition,
        minimum_total_energy=minimum_total,
        minimum_family_energy=minimum_family,
    )


def causal_normalized_arrival_energy(
    times_seconds: np.ndarray,
    history: CausalPositiveBandEnergyHistory,
    *,
    initial_source_energy: np.ndarray,
    window_seconds: tuple[float, float],
) -> CausalNormalizedArrivalEnergy:
    """Integrate positive receiving-band energy over one frozen window."""

    times = np.asarray(times_seconds, dtype=float)
    initial = np.asarray(initial_source_energy, dtype=float).ravel()
    total = np.asarray(history.total_energy, dtype=float)
    family = np.asarray(history.family_energy, dtype=float)
    lower, upper = (float(item) for item in window_seconds)
    if (
        times.ndim != 1
        or total.shape != (times.size, initial.size)
        or family.shape[:2] != total.shape
        or np.any(~np.isfinite(initial))
        or np.any(initial <= 0.0)
        or not lower < upper
    ):
        raise ValueError("normalized arrival-energy inputs are invalid")
    duration = upper - lower
    total_average = (
        causal_integrate_frozen_window(times, total, window_seconds)
        / duration
        / initial
    )
    family_average = (
        causal_integrate_frozen_window(times, family, window_seconds)
        / duration
        / initial[:, None]
    )
    normalized_history = total / initial[None, :]
    peak = np.max(normalized_history, axis=0)
    partition_scale = max(
        float(np.max(np.abs(total_average))),
        float(np.max(np.abs(family_average))),
        np.finfo(float).tiny,
    )
    partition = float(
        np.max(
            np.abs(np.sum(family_average, axis=-1) - total_average)
        )
        / partition_scale
    )
    return CausalNormalizedArrivalEnergy(
        total_time_average=np.asarray(total_average),
        family_time_average=np.asarray(family_average),
        peak_total=np.asarray(peak),
        maximum_integrated_partition_relative_defect=partition,
    )
