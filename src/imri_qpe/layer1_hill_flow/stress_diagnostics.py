"""Shock/Reynolds stress diagnostics for Hill-flow simulation outputs."""

from __future__ import annotations

import numpy as np


def _integrate_vertical(quantity, dz):
    quantity = np.asarray(quantity, dtype=float)
    dz = np.asarray(dz, dtype=float)
    if quantity.ndim == 0:
        return quantity * dz
    return np.sum(quantity * dz, axis=-1)


def _scalar_or_array(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array


def reynolds_stress(rho, dv_R, dv_phi, dz):
    """Compute W_Rphi = int rho delta_v_R delta_v_phi dz."""

    integrand = np.asarray(rho, dtype=float) * np.asarray(dv_R, dtype=float) * np.asarray(dv_phi, dtype=float)
    return _scalar_or_array(_integrate_vertical(integrand, dz))


def alpha_shock(W_Rphi, P, dz):
    """Compute alpha_shock = W_Rphi / int P dz."""

    W_Rphi = np.asarray(W_Rphi, dtype=float)
    pressure_column = np.asarray(_integrate_vertical(P, dz), dtype=float)
    result = np.divide(
        W_Rphi,
        pressure_column,
        out=np.full(np.broadcast(W_Rphi, pressure_column).shape, np.nan),
        where=pressure_column != 0.0,
    )
    return _scalar_or_array(result)
