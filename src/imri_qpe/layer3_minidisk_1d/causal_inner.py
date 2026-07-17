"""Causal thermodynamic and characteristic diagnostics for an inner flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C, DEFAULT_MU_MOL

from .entropy_advection import (
    gas_radiation_adiabatic_sound_speed_squared,
    gas_radiation_specific_enthalpy,
)


def _as_float_or_array(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array


def gas_radiation_relativistic_sound_speed_squared(
    rho,
    T,
    mu_mol: float = DEFAULT_MU_MOL,
    gamma_gas: float = 5.0 / 3.0,
):
    """Return the causal sound speed for the shared gas+radiation EOS.

    The total energy density includes rest mass. Along an adiabat,

    ``d epsilon / d rho = c^2 + e + P/rho``.

    Therefore ``a^2 = c^2 (dP/d rho)_s / (d epsilon/d rho)_s``. This is a
    thermodynamic derivative, not a numerical cap on the Newtonian speed.
    """

    if not 1.0 < gamma_gas <= 2.0:
        raise ValueError("relativistic gas gamma must lie in (1, 2]")
    newtonian = np.asarray(
        gas_radiation_adiabatic_sound_speed_squared(
            rho,
            T,
            mu_mol=mu_mol,
            gamma_gas=gamma_gas,
        ),
        dtype=float,
    )
    thermal_enthalpy = np.asarray(
        gas_radiation_specific_enthalpy(
            rho,
            T,
            mu_mol=mu_mol,
            gamma_gas=gamma_gas,
        ),
        dtype=float,
    )
    sound_speed_squared = C**2 * newtonian / (C**2 + thermal_enthalpy)
    if np.any(~np.isfinite(sound_speed_squared)) or np.any(
        sound_speed_squared <= 0.0
    ):
        raise ValueError("relativistic sound speed is not positive and finite")
    if np.any(sound_speed_squared >= C**2):
        raise ValueError("relativistic sound speed is not subluminal")
    return _as_float_or_array(sound_speed_squared)


def special_relativistic_radial_characteristic_speeds(
    radial_velocity: float,
    sound_speed: float,
) -> tuple[float, float, float, float]:
    """Return one-dimensional radial SR speeds in outward orientation.

    This prototype does not include relativistic transverse-velocity effects.
    """

    velocity = float(radial_velocity)
    sound = float(sound_speed)
    if not np.isfinite(velocity) or abs(velocity) >= C:
        raise ValueError("radial velocity must be finite and subluminal")
    if not np.isfinite(sound) or not 0.0 < sound < C:
        raise ValueError("sound speed must be finite, positive, and subluminal")
    ratio = velocity * sound / C**2
    acoustic_minus = (velocity - sound) / (1.0 - ratio)
    acoustic_plus = (velocity + sound) / (1.0 + ratio)
    speeds = (acoustic_minus, velocity, velocity, acoustic_plus)
    if any(not np.isfinite(value) or abs(value) >= C for value in speeds):
        raise ValueError("relativistic characteristic speed is not causal")
    return tuple(float(value) for value in speeds)


@dataclass(frozen=True)
class CausalInnerCharacteristicAudit:
    """Local orthonormal characteristic count at an inner radial boundary."""

    radial_velocity: float
    sound_speed: float
    radial_mach_number: float
    characteristic_speeds: tuple[float, float, float, float]
    incoming_characteristics: int

    @property
    def causally_outgoing(self) -> bool:
        return self.incoming_characteristics == 0


def audit_causal_inner_characteristics(
    radial_velocity: float,
    sound_speed: float,
) -> CausalInnerCharacteristicAudit:
    """Count local characteristics entering through the inner boundary."""

    speeds = special_relativistic_radial_characteristic_speeds(
        radial_velocity,
        sound_speed,
    )
    return CausalInnerCharacteristicAudit(
        radial_velocity=float(radial_velocity),
        sound_speed=float(sound_speed),
        radial_mach_number=float(radial_velocity / sound_speed),
        characteristic_speeds=speeds,
        incoming_characteristics=sum(value > 0.0 for value in speeds),
    )
