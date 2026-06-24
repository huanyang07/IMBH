"""Small cgs unit-conversion helpers."""

from __future__ import annotations

from .constants import C, G, JULIAN_YEAR, M_SUN


def solar_masses_to_g(mass_msun: float) -> float:
    """Convert solar masses to grams."""

    return mass_msun * M_SUN


def g_to_solar_masses(mass_g: float) -> float:
    """Convert grams to solar masses."""

    return mass_g / M_SUN


def years_to_seconds(time_yr: float) -> float:
    """Convert Julian years to seconds."""

    return time_yr * JULIAN_YEAR


def seconds_to_years(time_s: float) -> float:
    """Convert seconds to Julian years."""

    return time_s / JULIAN_YEAR


def msun_per_year_to_g_per_s(mdot_msun_yr: float) -> float:
    """Convert an accretion rate from solar masses per year to g/s."""

    return solar_masses_to_g(mdot_msun_yr) / JULIAN_YEAR


def g_per_s_to_msun_per_year(mdot_g_s: float) -> float:
    """Convert an accretion rate from g/s to solar masses per year."""

    return mdot_g_s * JULIAN_YEAR / M_SUN


def gravitational_radius(M_g: float) -> float:
    """Return r_g = GM/c^2 for mass ``M_g``."""

    if M_g <= 0.0:
        raise ValueError("M_g must be positive")
    return G * M_g / C**2

