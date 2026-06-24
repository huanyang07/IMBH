"""Reduced one-zone relaxation oscillator from the project note."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OneZoneCycleResult:
    """Derived one-zone cycle quantities in cgs units."""

    M_min: float
    delta_M: float
    t_load: float
    t_high: float
    P_QPE: float
    duty_cycle: float
    mdot_burst: float
    mdot_net_load: float


def one_zone_cycle(
    Mmax: float,
    zeta: float,
    mdot_cap: float,
    mdot_low: float,
    alpha_hot: float,
    H_over_R_hot: float,
    Omega_K: float,
    t_trans: float = 0.0,
) -> OneZoneCycleResult:
    """Return the reduced Layer-3 one-zone limit-cycle estimates.

    Inputs are cgs: masses in g, rates in g/s, frequencies in s^-1, and
    times in seconds.
    """

    if Mmax <= 0.0:
        raise ValueError("Mmax must be positive")
    if not 0.0 < zeta < 1.0:
        raise ValueError("zeta must be between 0 and 1")
    if mdot_cap <= mdot_low:
        raise ValueError("mdot_cap must exceed mdot_low for loading")
    if alpha_hot <= 0.0:
        raise ValueError("alpha_hot must be positive")
    if H_over_R_hot <= 0.0:
        raise ValueError("H_over_R_hot must be positive")
    if Omega_K <= 0.0:
        raise ValueError("Omega_K must be positive")
    if t_trans < 0.0:
        raise ValueError("t_trans must be non-negative")

    M_min = zeta * Mmax
    delta_M = Mmax - M_min
    mdot_net_load = mdot_cap - mdot_low
    t_load = delta_M / mdot_net_load
    t_high = 1.0 / (alpha_hot * H_over_R_hot**2 * Omega_K)
    P_QPE = t_load + t_trans + t_high
    duty_cycle = t_high / P_QPE
    mdot_burst = delta_M / t_high

    return OneZoneCycleResult(
        M_min=M_min,
        delta_M=delta_M,
        t_load=t_load,
        t_high=t_high,
        P_QPE=P_QPE,
        duty_cycle=duty_cycle,
        mdot_burst=mdot_burst,
        mdot_net_load=mdot_net_load,
    )

