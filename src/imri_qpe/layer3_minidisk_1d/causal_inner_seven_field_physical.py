"""Candidate physical seven-field causal inner-column closure.

This module contains the smooth local state and flux map used by the
seven-field structural audit.  It is deliberately separate from the
certified five-field trajectory implementation: no spatial discretization
or time step dispatches through this file.

The primitive chart is

``(ln Sigma, beta_R, beta_phi, ln T, chi, ln H, beta_H)``.

The shear reservoir coefficient is not fitted.  It is the positive solution
of ``a_pi = h_ext c_nu**2`` with
``c_nu**2 = alpha c_s**2/c**2`` and with the same reservoir included in
``h_ext``.  Thus the quadratic shear energy and the Maxwell--Cattaneo signal
calibration use one coefficient rather than two independently adjustable
ones.

The map below is a candidate to be audited, not a certified production
closure.  In particular, the structural package must reject it if the
entropy-flux one-form is not integrable or if its principal pencil is not
symmetric hyperbolic on the prospectively frozen envelope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C, DEFAULT_MU_MOL

from .causal_inner_geometry import (
    KerrSchildColumnGeometry,
    ValenciaPerfectFluidPrimitive,
)
from .causal_inner_recovery import (
    FixedHeightGasRadiationColumnEOS,
    GasRadiationColumnThermodynamics,
)
from .causal_inner_stress import causal_stress_column_state
from .entropy_advection import gas_radiation_specific_entropy


SEVEN_FIELD_PHYSICAL_PRIMITIVE_NAMES = (
    "log_surface_density",
    "radial_velocity_over_c",
    "azimuthal_velocity_over_c",
    "log_temperature",
    "specific_shear_stress",
    "log_proper_half_thickness",
    "vertical_velocity_over_c",
)


@dataclass(frozen=True)
class SevenFieldRelaxationCalibration:
    """State-local coefficients fixed by the EOS and alpha prescription."""

    viscous_signal_speed_over_c: float
    reservoir_coefficient: float
    thermal_specific_enthalpy_over_c2: float
    extended_specific_enthalpy_over_c2: float
    vertical_specific_energy_over_c2: float
    shear_specific_energy_over_c2: float
    equilibrium_specific_stress: float
    specific_shear_viscosity_seconds: float
    relaxation_time_seconds: float
    vertical_damping_rate_per_second: float


@dataclass(frozen=True)
class SevenFieldPhysicalState:
    """One candidate seven-field state, flux, entropy, and calibration."""

    primitive_chart: np.ndarray
    thermodynamics: GasRadiationColumnThermodynamics
    calibration: SevenFieldRelaxationCalibration
    conserved: np.ndarray
    flux_over_c: np.ndarray
    mathematical_entropy: float
    physical_entropy_density: float
    proper_vertical_frequency: float


def _require_chart(chart) -> np.ndarray:
    values = np.asarray(chart, dtype=float)
    if values.shape != (7,) or np.any(~np.isfinite(values)):
        raise ValueError("seven-field primitive chart must be finite and length seven")
    if values[1] ** 2 + values[2] ** 2 >= 1.0:
        raise ValueError("horizontal velocity must be subluminal")
    if abs(float(values[6])) >= 1.0:
        raise ValueError("vertical velocity must be subluminal")
    return values


def _positive_shear_reservoir_coefficient(
    *,
    signal_speed_squared_over_c2: float,
    enthalpy_without_shear_over_c2: float,
    specific_stress: float,
) -> float:
    """Solve ``a=h_ext*c_nu^2`` including ``chi^2/(2a)`` in ``h_ext``."""

    k = float(signal_speed_squared_over_c2)
    h0 = float(enthalpy_without_shear_over_c2)
    chi = float(specific_stress)
    if not 0.0 < k < 1.0 or not np.isfinite(h0) or h0 <= 0.0:
        raise ValueError("shear reservoir calibration is not physical")
    discriminant = (k * h0) ** 2 + 2.0 * k * chi**2
    coefficient = 0.5 * (k * h0 + np.sqrt(discriminant))
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("shear reservoir coefficient is not positive")
    return float(coefficient)


def seven_field_physical_state(
    geometry: KerrSchildColumnGeometry,
    chart,
    *,
    proper_vertical_frequency: float,
    alpha: float,
    stress_factor: float = 1.0,
    mu_mol: float = DEFAULT_MU_MOL,
    gamma_gas: float = 5.0 / 3.0,
) -> SevenFieldPhysicalState:
    """Evaluate the candidate Kerr--Schild seven-field state and radial flux."""

    values = _require_chart(chart)
    omega = float(proper_vertical_frequency)
    alpha = float(alpha)
    stress_factor = float(stress_factor)
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("proper vertical frequency must be positive")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not np.isfinite(stress_factor) or stress_factor <= 0.0:
        raise ValueError("stress factor must be positive")

    sigma = float(np.exp(values[0]))
    beta_r = float(values[1])
    beta_phi = float(values[2])
    temperature = float(np.exp(values[3]))
    chi = float(values[4])
    height = float(np.exp(values[5]))
    beta_h = float(values[6])
    eos = FixedHeightGasRadiationColumnEOS(
        proper_half_thickness=height,
        mu_mol=mu_mol,
        gamma_gas=gamma_gas,
    )
    thermodynamics = eos.from_surface_density_temperature(
        sigma,
        temperature,
    )
    thermal_enthalpy = (
        1.0
        + thermodynamics.specific_internal_energy / C**2
        + thermodynamics.integrated_pressure / (sigma * C**2)
    )
    vertical_energy = 0.5 * (
        beta_h**2 + (omega * height / C) ** 2
    )
    enthalpy_without_shear = thermal_enthalpy + vertical_energy
    signal_speed = np.sqrt(alpha) * thermodynamics.sound_speed / C
    reservoir_coefficient = _positive_shear_reservoir_coefficient(
        signal_speed_squared_over_c2=signal_speed**2,
        enthalpy_without_shear_over_c2=enthalpy_without_shear,
        specific_stress=chi,
    )
    shear_energy = chi**2 / (2.0 * reservoir_coefficient)
    extended_enthalpy = enthalpy_without_shear + shear_energy
    calibration_defect = abs(
        reservoir_coefficient - extended_enthalpy * signal_speed**2
    )
    if calibration_defect > 2.0e-14 * max(reservoir_coefficient, 1.0):
        raise RuntimeError("implicit shear reservoir calibration did not close")

    equilibrium_stress = (
        stress_factor
        * alpha
        * thermodynamics.integrated_pressure
        / (sigma * C**2)
    )
    reference_shear = 1.5 * omega
    specific_viscosity = equilibrium_stress / reference_shear
    relaxation_time = specific_viscosity / reservoir_coefficient
    if relaxation_time <= 0.0 or not np.isfinite(relaxation_time):
        raise ValueError("shear relaxation time is not positive")

    augmented_internal_energy = thermodynamics.specific_internal_energy + C**2 * (
        vertical_energy + shear_energy
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=sigma,
        radial_velocity_over_c=beta_r,
        azimuthal_velocity_over_c=beta_phi,
        specific_internal_energy=float(augmented_internal_energy),
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    stress = causal_stress_column_state(
        geometry,
        primitive,
        specific_stress=chi,
    )
    rest_mass = float(stress.killing_conserved[0])
    transport = float(stress.stress_killing_flux_increment_over_c[0])
    # The stress increment has zero mass flux.  Use the perfect-fluid
    # transport velocity recovered from the relaxing coordinate instead.
    lorentz = 1.0 / np.sqrt(1.0 - beta_r**2 - beta_phi**2)
    transport_velocity = (
        geometry.base.lapse * beta_r / np.sqrt(geometry.base.gamma_rr)
        - geometry.base.radial_shift_over_c
    )
    if abs(rest_mass - sigma * lorentz) > 2.0e-13 * max(rest_mass, 1.0):
        raise RuntimeError("seven-field rest-mass map is inconsistent")
    del transport

    relaxing_stress_state = rest_mass * chi
    relaxing_stress_flux = (
        relaxing_stress_state * transport_velocity
        + geometry.base.lapse
        * rest_mass
        * reservoir_coefficient
        * beta_phi
        / np.sqrt(geometry.base.gamma_rr)
    )
    height_state = rest_mass * height
    height_flux = height_state * transport_velocity
    vertical_velocity = C * beta_h
    vertical_momentum_state = rest_mass * vertical_velocity
    vertical_momentum_flux = vertical_momentum_state * transport_velocity
    conserved = np.concatenate(
        (
            stress.killing_conserved,
            np.asarray(
                [
                    relaxing_stress_state,
                    height_state,
                    vertical_momentum_state,
                ],
                dtype=float,
            ),
        )
    )
    flux = np.concatenate(
        (
            stress.killing_flux_over_c,
            np.asarray(
                [
                    relaxing_stress_flux,
                    height_flux,
                    vertical_momentum_flux,
                ],
                dtype=float,
            ),
        )
    )
    specific_entropy = gas_radiation_specific_entropy(
        thermodynamics.density,
        temperature,
        mu_mol=mu_mol,
        gamma_gas=gamma_gas,
    )
    physical_entropy = rest_mass * float(specific_entropy)
    calibration = SevenFieldRelaxationCalibration(
        viscous_signal_speed_over_c=float(signal_speed),
        reservoir_coefficient=reservoir_coefficient,
        thermal_specific_enthalpy_over_c2=float(thermal_enthalpy),
        extended_specific_enthalpy_over_c2=float(extended_enthalpy),
        vertical_specific_energy_over_c2=float(vertical_energy),
        shear_specific_energy_over_c2=float(shear_energy),
        equilibrium_specific_stress=float(equilibrium_stress),
        specific_shear_viscosity_seconds=float(specific_viscosity),
        relaxation_time_seconds=float(relaxation_time),
        vertical_damping_rate_per_second=float(alpha * omega),
    )
    return SevenFieldPhysicalState(
        primitive_chart=np.array(values, copy=True),
        thermodynamics=thermodynamics,
        calibration=calibration,
        conserved=np.asarray(conserved, dtype=float),
        flux_over_c=np.asarray(flux, dtype=float),
        mathematical_entropy=float(-physical_entropy),
        physical_entropy_density=float(physical_entropy),
        proper_vertical_frequency=omega,
    )
