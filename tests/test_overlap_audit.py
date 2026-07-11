from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.overlap_audit import (
    OverlapGateConfig,
    contiguous_passing_bands,
    effective_optical_depth,
    intersect_bands,
    kramers_absorption_opacity,
    overlap_diagnostics,
)
from imri_qpe.units import gravitational_radius, solar_masses_to_g


def test_absorption_and_effective_depth_are_positive_and_bracketed() -> None:
    rho = np.asarray([1.0e-6, 2.0e-6])
    temperature = np.asarray([1.0e6, 2.0e6])
    low = kramers_absorption_opacity(rho, temperature)
    high = kramers_absorption_opacity(rho, temperature, coefficient=5.0e24)
    assert np.all(high > low)
    assert np.all(effective_optical_depth(high, 20.0) > effective_optical_depth(low, 20.0))


def test_overlap_bands_split_on_failed_gate() -> None:
    mass = solar_masses_to_g(1.0e4)
    radius = np.geomspace(10.0, 100.0, 32) * gravitational_radius(mass)
    sigma = 1.0e5 * (radius / radius[0])**-0.2
    temperature = 1.0e6 * (radius / radius[0])**-0.4
    H = 0.1 * radius
    pressure = np.full(radius.size, 0.01)
    pressure[15] = 0.2
    diagnostics = overlap_diagnostics(
        radius, sigma, temperature, H, sigma / (2.0 * H),
        np.full(radius.size, -1.0e-4), pressure, mass,
        tau_scattering=np.full(radius.size, 100.0),
        config=OverlapGateConfig(min_tau_effective=0.0),
    )
    bands = contiguous_passing_bands(diagnostics)
    assert len(bands) == 2
    assert bands[0][1] < radius[15] < bands[1][0]


def test_intersect_bands_keeps_only_common_ranges() -> None:
    assert intersect_bands(
        [(12.0, 20.0), (30.0, 60.0)],
        [(15.0, 35.0), (50.0, 70.0)],
    ) == [(15.0, 20.0), (30.0, 35.0), (50.0, 60.0)]
