"""Focused tests for one-way scattering history helpers."""

from __future__ import annotations

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d.causal_inner_one_way_scattering import (
    causal_amplitude_scaling_defect,
    causal_integrate_frozen_window,
    causal_integrated_one_way_ledger,
    causal_one_way_energy_history,
)


def test_frozen_window_interpolates_endpoints() -> None:
    times = np.asarray((0.0, 1.0, 2.0))
    values = np.column_stack((times, times**2))
    integrated = causal_integrate_frozen_window(
        times,
        values,
        (0.5, 1.5),
    )
    assert np.allclose(integrated, (1.0, 1.25))


def test_one_way_energy_ledger_closes_with_explicit_discrete_remainder() -> None:
    times = np.linspace(0.0, 1.0, 9)
    cells = 6
    cases = 2
    history = np.zeros((times.size, cases, cells, 5))
    history[:, 0, :, 0] = np.exp(-times[:, None])
    history[:, 1] = 0.5 * history[:, 0]
    energy = np.repeat(np.eye(5)[None, :, :], cells, axis=0)
    flux = -np.repeat(np.eye(5)[None, :, :], cells, axis=0)
    projectors = np.zeros((cells, 5, 5, 5))
    for family in range(5):
        projectors[:, family, family, family] = 1.0
    lower = {"stress_relaxation": np.zeros_like(energy)}
    evaluated = causal_one_way_energy_history(
        history,
        log_edges=np.linspace(0.0, 1.0, cells + 1),
        energy_metrics=energy,
        flux_metrics=flux,
        projectors=projectors,
        lower_evolution_blocks=lower,
        downstream_face=1,
        interface_face=5,
    )
    assert np.all(evaluated.incident_total_flux >= 0.0)
    assert np.allclose(
        np.sum(evaluated.incident_family_fluxes, axis=-1),
        evaluated.incident_total_flux,
    )
    assert np.isclose(
        evaluated.incident_total_flux[0, 0],
        0.5 * C,
    )
    ledger = causal_integrated_one_way_ledger(
        evaluated,
        times,
        incident_window_seconds=(0.0, 1.0),
        transmitted_window_seconds=(0.0, 1.0),
    )
    assert ledger.maximum_relative_ledger_defect <= 1.0e-15
    assert np.allclose(ledger.transmission, 1.0)
    assert np.allclose(ledger.family_transmission[:, 0], 1.0)
    assert np.allclose(ledger.family_transmission[:, 1:], 0.0)


def test_amplitude_scaling_defect() -> None:
    reference = np.asarray((1.0, -2.0, 3.0))
    assert causal_amplitude_scaling_defect(
        reference,
        -0.5 * reference,
        -0.5,
    ) == 0.0
