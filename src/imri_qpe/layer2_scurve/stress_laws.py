"""Stress prescriptions for the local thermal-equilibrium solver."""

from __future__ import annotations

import numpy as np


def _as_float_or_array(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array


def stress_pressure(Pgas, Ptot, alpha: float, mu_stress: float):
    """Return alpha * Pgas^mu_stress * Ptot^(1 - mu_stress)."""

    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if not 0.0 <= mu_stress <= 1.0:
        raise ValueError("mu_stress must be between 0 and 1")

    Pgas = np.asarray(Pgas, dtype=float)
    Ptot = np.asarray(Ptot, dtype=float)
    if np.any(Pgas < 0.0) or np.any(Ptot < 0.0):
        raise ValueError("pressures must be non-negative")

    return _as_float_or_array(alpha * Pgas**mu_stress * Ptot ** (1.0 - mu_stress))


def instability_mu_criterion(mu_stress: float) -> bool:
    """Return True when the radiation-pressure branch is thermally unstable."""

    return mu_stress < 4.0 / 7.0

