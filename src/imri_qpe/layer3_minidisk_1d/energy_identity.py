"""One-zone pressure-work identities for conservative radial energy fluxes."""

from __future__ import annotations

import numpy as np


def _float_or_array(value):
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError("energy-identity inputs must be finite")
    if array.ndim == 0:
        return float(array)
    return array


def enthalpy_vertical_work(
    mdot,
    surface_density,
    integrated_pressure,
    surface_density_derivative,
    pressure,
    density,
    density_derivative,
):
    """Return work paired with ``Mdot*(e + Pi/Sigma)`` transport.

    Derivatives may be differential derivatives or finite cell increments, as
    long as both use the same radial coordinate. In the one-zone closure this
    is equivalently ``Mdot*(P/rho)*dln(H)``.
    """

    sigma = np.asarray(surface_density, dtype=float)
    rho = np.asarray(density, dtype=float)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("surface_density must be positive and finite")
    if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
        raise ValueError("density must be positive and finite")
    result = np.asarray(mdot, dtype=float) * (
        np.asarray(integrated_pressure, dtype=float)
        * np.asarray(surface_density_derivative, dtype=float)
        / sigma**2
        - np.asarray(pressure, dtype=float)
        * np.asarray(density_derivative, dtype=float)
        / rho**2
    )
    return _float_or_array(result)


def internal_energy_vertical_work(
    mdot,
    surface_density,
    integrated_pressure_derivative,
    pressure,
    density,
    density_derivative,
):
    """Return work paired with ``Mdot*e`` rather than enthalpy transport."""

    sigma = np.asarray(surface_density, dtype=float)
    rho = np.asarray(density, dtype=float)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("surface_density must be positive and finite")
    if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
        raise ValueError("density must be positive and finite")
    result = np.asarray(mdot, dtype=float) * (
        np.asarray(integrated_pressure_derivative, dtype=float) / sigma
        - np.asarray(pressure, dtype=float)
        * np.asarray(density_derivative, dtype=float)
        / rho**2
    )
    return _float_or_array(result)
