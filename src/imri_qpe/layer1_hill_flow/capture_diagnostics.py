"""Mass and angular-momentum diagnostics for Hill-sphere capture."""

from __future__ import annotations

import numpy as np


def _scalar_or_array(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array


def capture_fraction(mdot_cap, mdot_leak):
    """Return f_cap = Mdot_cap / Mdot_leak, with zero-safe division."""

    cap = np.asarray(mdot_cap, dtype=float)
    leak = np.asarray(mdot_leak, dtype=float)
    result = np.divide(cap, leak, out=np.zeros(np.broadcast(cap, leak).shape), where=leak != 0.0)
    return _scalar_or_array(result)


def mean_specific_angular_momentum(mass_flux, angular_momentum_flux) -> float:
    """Return total angular-momentum flux divided by total mass flux."""

    mass_flux = np.asarray(mass_flux, dtype=float)
    angular_momentum_flux = np.asarray(angular_momentum_flux, dtype=float)
    total_mass_flux = np.sum(mass_flux)
    if total_mass_flux == 0.0:
        return 0.0
    return float(np.sum(angular_momentum_flux) / total_mass_flux)


def recycling_time(M_H, mdot_in, mdot_cap):
    """Estimate t_recyc = M_H / (mdot_in - mdot_cap).

    Non-positive recycling rates return ``inf`` because a steady recycling
    reservoir cannot be inferred from that mass budget.
    """

    mass = np.asarray(M_H, dtype=float)
    rate = np.asarray(mdot_in, dtype=float) - np.asarray(mdot_cap, dtype=float)
    result = np.divide(mass, rate, out=np.full(np.broadcast(mass, rate).shape, np.inf), where=rate > 0.0)
    return _scalar_or_array(result)

