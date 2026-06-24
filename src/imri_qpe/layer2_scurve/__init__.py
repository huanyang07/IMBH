"""Layer-2 vertical-equilibrium and S-curve tools."""

from .stress_laws import instability_mu_criterion, stress_pressure
from .thermal_equilibrium import (
    ThermalEquilibriumParams,
    accretion_rate_from_heating,
    equilibrium_residual,
    q_adv_minus,
    q_plus_alpha,
    q_rad_minus,
    q_wind_minus,
    temperature_activated_cooling_fraction,
    thermal_equilibrium_quantities,
)
from .turning_points import ScurveResult, compute_scurve, find_turning_points
from .vertical_structure import (
    VerticalState,
    effective_temperature,
    gas_pressure,
    midplane_density,
    optical_depth,
    radiation_pressure,
    solve_scale_height,
    vertical_state,
)

__all__ = [
    "ScurveResult",
    "ThermalEquilibriumParams",
    "VerticalState",
    "accretion_rate_from_heating",
    "compute_scurve",
    "effective_temperature",
    "equilibrium_residual",
    "find_turning_points",
    "gas_pressure",
    "instability_mu_criterion",
    "midplane_density",
    "optical_depth",
    "q_plus_alpha",
    "q_adv_minus",
    "q_rad_minus",
    "q_wind_minus",
    "radiation_pressure",
    "solve_scale_height",
    "stress_pressure",
    "temperature_activated_cooling_fraction",
    "thermal_equilibrium_quantities",
    "vertical_state",
]
