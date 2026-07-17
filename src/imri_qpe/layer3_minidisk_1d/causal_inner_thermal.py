"""Dynamic column height and thermal sources for the causal inner flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from imri_qpe.constants import (
    A_RAD,
    C,
    DEFAULT_KAPPA_ES,
    DEFAULT_MU_MOL,
    SIGMA_SB,
)
from imri_qpe.scales import gas_constant_per_gram

from .causal_inner_geometry import (
    KerrSchildColumnGeometry,
    ValenciaPerfectFluidPrimitive,
)
from .causal_inner_recovery import GasRadiationColumnThermodynamics


@dataclass(frozen=True)
class QuasiHydrostaticColumnDerivatives:
    """Logarithmic thermodynamic derivatives of the responsive column."""

    height_log_surface_density: float
    height_log_temperature: float
    height_log_vertical_frequency: float
    density_log_surface_density: float
    density_log_temperature: float
    internal_energy_log_surface_density: float
    internal_energy_log_temperature: float
    pressure_log_surface_density: float
    pressure_log_temperature: float
    adiabatic_log_temperature_log_surface_density: float
    adiabatic_integrated_pressure_derivative: float
    sound_speed_over_c: float


@dataclass(frozen=True)
class QuasiHydrostaticCharacteristicAudit:
    """Local-rest characteristic audit including vertical pressure work."""

    analytic_speeds_over_c: tuple[float, float, float]
    numerical_speeds_over_c: tuple[float, float, float]
    maximum_eigenvalue_defect: float
    maximum_imaginary_eigenvalue: float


def _quasi_hydrostatic_derivatives(
    *,
    sigma: float,
    temperature: float,
    height: float,
    density: float,
    pressure: float,
    integrated_pressure: float,
    specific_internal_energy: float,
    proper_vertical_frequency: float,
    mu_mol: float,
    gamma_gas: float,
) -> QuasiHydrostaticColumnDerivatives:
    """Return analytic state and physical-adiabat derivatives."""

    gas_constant = gas_constant_per_gram(mu_mol)
    radiation_energy_density = A_RAD * temperature**4
    radiation_height_term = (
        2.0 * radiation_energy_density * height / (3.0 * sigma)
    )
    orbital_height_term = proper_vertical_frequency**2 * height**2
    derivative_denominator = (
        2.0 * orbital_height_term - radiation_height_term
    )
    height_sigma = -radiation_height_term / derivative_denominator
    height_temperature = (
        4.0 * radiation_height_term + gas_constant * temperature
    ) / derivative_denominator
    height_vertical_frequency = (
        -2.0 * orbital_height_term / derivative_denominator
    )
    density_sigma = 1.0 - height_sigma
    density_temperature = -height_temperature

    gas_internal = (
        gas_constant * temperature / (gamma_gas - 1.0)
    )
    radiation_internal = radiation_energy_density / density
    internal_sigma = -radiation_internal * density_sigma
    internal_temperature = (
        gas_internal
        + radiation_internal * (4.0 - density_temperature)
    )

    gas_integrated_pressure = sigma * gas_constant * temperature
    radiation_integrated_pressure = (
        2.0 * height * radiation_energy_density / 3.0
    )
    pressure_sigma = (
        gas_integrated_pressure
        + radiation_integrated_pressure * height_sigma
    )
    pressure_temperature = (
        gas_integrated_pressure
        + radiation_integrated_pressure
        * (4.0 + height_temperature)
    )

    pressure_over_density = pressure / density
    entropy_sigma = (
        internal_sigma - pressure_over_density * density_sigma
    )
    entropy_temperature = (
        internal_temperature
        - pressure_over_density * density_temperature
    )
    adiabatic_temperature = -entropy_sigma / entropy_temperature
    adiabatic_pressure_log = (
        pressure_sigma
        + pressure_temperature * adiabatic_temperature
    )
    adiabatic_pressure_derivative = adiabatic_pressure_log / sigma
    enthalpy_over_c2 = (
        1.0
        + specific_internal_energy / C**2
        + integrated_pressure / (sigma * C**2)
    )
    sound_squared_over_c2 = (
        adiabatic_pressure_derivative / (enthalpy_over_c2 * C**2)
    )
    if not 0.0 < sound_squared_over_c2 < 1.0:
        raise ValueError("responsive-column acoustic speed is not causal")
    return QuasiHydrostaticColumnDerivatives(
        height_log_surface_density=float(height_sigma),
        height_log_temperature=float(height_temperature),
        height_log_vertical_frequency=float(
            height_vertical_frequency
        ),
        density_log_surface_density=float(density_sigma),
        density_log_temperature=float(density_temperature),
        internal_energy_log_surface_density=float(internal_sigma),
        internal_energy_log_temperature=float(internal_temperature),
        pressure_log_surface_density=float(pressure_sigma),
        pressure_log_temperature=float(pressure_temperature),
        adiabatic_log_temperature_log_surface_density=float(
            adiabatic_temperature
        ),
        adiabatic_integrated_pressure_derivative=float(
            adiabatic_pressure_derivative
        ),
        sound_speed_over_c=float(np.sqrt(sound_squared_over_c2)),
    )


@dataclass(frozen=True)
class QuasiHydrostaticGasRadiationColumnEOS:
    """Gas+radiation column with algebraically responsive proper height."""

    proper_vertical_frequency: float
    mu_mol: float = DEFAULT_MU_MOL
    gamma_gas: float = 5.0 / 3.0
    minimum_temperature: float = 1.0e-3
    maximum_temperature: float = 1.0e13

    def __post_init__(self) -> None:
        values = (
            self.proper_vertical_frequency,
            self.mu_mol,
            self.minimum_temperature,
            self.maximum_temperature,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("dynamic column EOS parameters must be finite")
        if self.proper_vertical_frequency <= 0.0 or self.mu_mol <= 0.0:
            raise ValueError("vertical frequency and molecular weight must be positive")
        if not 1.0 < self.gamma_gas <= 2.0:
            raise ValueError("gas gamma must lie in (1, 2]")
        if not (
            0.0 < self.minimum_temperature < self.maximum_temperature
        ):
            raise ValueError("column temperature bounds are invalid")

    def from_surface_density_temperature(
        self,
        surface_density: float,
        temperature: float,
    ) -> GasRadiationColumnThermodynamics:
        """Return the one-zone hydrostatic gas+radiation column."""

        sigma = float(surface_density)
        temperature = float(temperature)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("surface density must be positive and finite")
        if (
            not np.isfinite(temperature)
            or not self.minimum_temperature
            <= temperature
            <= self.maximum_temperature
        ):
            raise ValueError("temperature lies outside the column EOS bounds")
        gas_constant = gas_constant_per_gram(self.mu_mol)
        omega = float(self.proper_vertical_frequency)
        radiation_term = 2.0 * A_RAD * temperature**4 / (3.0 * sigma)
        height = (
            radiation_term
            + np.sqrt(
                radiation_term**2
                + 4.0 * omega**2 * gas_constant * temperature
            )
        ) / (2.0 * omega**2)
        density = sigma / (2.0 * height)
        gas_pressure = density * gas_constant * temperature
        radiation_pressure = A_RAD * temperature**4 / 3.0
        pressure = gas_pressure + radiation_pressure
        integrated_pressure = 2.0 * height * pressure
        internal_energy = (
            gas_constant * temperature / (self.gamma_gas - 1.0)
            + A_RAD * temperature**4 / density
        )
        specific_enthalpy = internal_energy + integrated_pressure / sigma

        derivatives = _quasi_hydrostatic_derivatives(
            sigma=sigma,
            temperature=temperature,
            height=float(height),
            density=float(density),
            pressure=float(pressure),
            integrated_pressure=float(integrated_pressure),
            specific_internal_energy=float(internal_energy),
            proper_vertical_frequency=omega,
            mu_mol=self.mu_mol,
            gamma_gas=self.gamma_gas,
        )
        return GasRadiationColumnThermodynamics(
            surface_density=sigma,
            temperature=temperature,
            proper_half_thickness=float(height),
            density=float(density),
            integrated_pressure=float(integrated_pressure),
            specific_internal_energy=float(internal_energy),
            specific_enthalpy=float(specific_enthalpy),
            sound_speed=float(derivatives.sound_speed_over_c * C),
        )

    def derivatives(
        self,
        surface_density: float,
        temperature: float,
    ) -> QuasiHydrostaticColumnDerivatives:
        """Return analytic derivatives including responsive vertical height."""

        state = self.from_surface_density_temperature(
            surface_density,
            temperature,
        )
        pressure = (
            state.integrated_pressure
            / (2.0 * state.proper_half_thickness)
        )
        return _quasi_hydrostatic_derivatives(
            sigma=state.surface_density,
            temperature=state.temperature,
            height=state.proper_half_thickness,
            density=state.density,
            pressure=pressure,
            integrated_pressure=state.integrated_pressure,
            specific_internal_energy=state.specific_internal_energy,
            proper_vertical_frequency=self.proper_vertical_frequency,
            mu_mol=self.mu_mol,
            gamma_gas=self.gamma_gas,
        )

    def from_surface_density_internal_energy(
        self,
        surface_density: float,
        specific_internal_energy: float,
    ) -> GasRadiationColumnThermodynamics:
        """Invert the responsive-height column energy for temperature."""

        sigma = float(surface_density)
        target = float(specific_internal_energy)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("surface density must be positive and finite")
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("specific internal energy must be positive and finite")

        def residual(log_temperature: float) -> float:
            temperature = float(
                np.clip(
                    np.exp(log_temperature),
                    self.minimum_temperature,
                    self.maximum_temperature,
                )
            )
            state = self.from_surface_density_temperature(
                sigma,
                temperature,
            )
            return state.specific_internal_energy - target

        lower = float(np.log(self.minimum_temperature))
        upper = float(np.log(self.maximum_temperature))
        if residual(lower) > 0.0 or residual(upper) < 0.0:
            raise ValueError("internal energy lies outside the dynamic EOS bounds")
        root = brentq(
            residual,
            lower,
            upper,
            xtol=1.0e-13,
            rtol=1.0e-13,
            maxiter=160,
        )
        return self.from_surface_density_temperature(
            sigma,
            float(np.exp(root)),
        )


def audit_quasi_hydrostatic_characteristics(
    eos: QuasiHydrostaticGasRadiationColumnEOS,
    *,
    surface_density: float,
    temperature: float,
) -> QuasiHydrostaticCharacteristicAudit:
    """Audit local-rest acoustic modes with vertical work in the principal part."""

    state = eos.from_surface_density_temperature(
        surface_density,
        temperature,
    )
    derivatives = eos.derivatives(surface_density, temperature)
    enthalpy_over_c2 = (
        1.0
        + state.specific_internal_energy / C**2
        + state.integrated_pressure
        / (state.surface_density * C**2)
    )
    mass_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, enthalpy_over_c2, 0.0],
            [
                derivatives.internal_energy_log_surface_density
                - (
                    state.integrated_pressure
                    / state.surface_density
                )
                * derivatives.density_log_surface_density,
                0.0,
                derivatives.internal_energy_log_temperature
                - (
                    state.integrated_pressure
                    / state.surface_density
                )
                * derivatives.density_log_temperature,
            ],
        ],
        dtype=float,
    )
    flux_matrix = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [
                derivatives.pressure_log_surface_density
                / (state.surface_density * C**2),
                0.0,
                derivatives.pressure_log_temperature
                / (state.surface_density * C**2),
            ],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    numerical = np.linalg.eigvals(
        np.linalg.solve(mass_matrix, flux_matrix)
    )
    order = np.argsort(np.real(numerical))
    numerical = numerical[order]
    analytic = np.asarray(
        [
            -derivatives.sound_speed_over_c,
            0.0,
            derivatives.sound_speed_over_c,
        ],
        dtype=float,
    )
    return QuasiHydrostaticCharacteristicAudit(
        analytic_speeds_over_c=tuple(float(value) for value in analytic),
        numerical_speeds_over_c=tuple(
            float(value) for value in np.real(numerical)
        ),
        maximum_eigenvalue_defect=float(
            np.max(np.abs(np.real(numerical) - analytic))
        ),
        maximum_imaginary_eigenvalue=float(
            np.max(np.abs(np.imag(numerical)))
        ),
    )


@dataclass(frozen=True)
class CausalComovingEnergySource:
    """One isotropic comoving energy exchange transformed to Killing form."""

    comoving_energy_rate: float
    four_force: np.ndarray
    killing_source_per_ct: np.ndarray
    recovered_comoving_rate: float
    relative_identity_defect: float
    comoving_momentum_relative_defect: float


@dataclass(frozen=True)
class CausalThermalColumnSource:
    """Cooling and vertical work in one non-double-counted source ledger."""

    thermodynamics: GasRadiationColumnThermodynamics
    scattering_optical_depth: float
    radiative_cooling_rate: float
    vertical_work_rate: float
    cooling_source: CausalComovingEnergySource
    vertical_work_source: CausalComovingEnergySource
    total_killing_source_per_ct: np.ndarray
    local_viscous_energy_source: float


@dataclass(frozen=True)
class StressWorkPartition:
    """Exact finite product rule for torque work."""

    torque_work_flux_difference: float
    angular_exchange_work: float
    shear_conversion_work: float
    product_rule_defect: float
    explicit_total_energy_heating_source: float


def _column_four_velocity(
    geometry: KerrSchildColumnGeometry,
    primitive: ValenciaPerfectFluidPrimitive,
) -> np.ndarray:
    """Return the contravariant four-velocity in ``(ct,R,phi)``."""

    beta_r = float(primitive.radial_velocity_over_c)
    beta_phi = float(primitive.azimuthal_velocity_over_c)
    speed_squared = beta_r**2 + beta_phi**2
    if speed_squared >= 1.0:
        raise ValueError("Eulerian three-velocity must be subluminal")
    lorentz = 1.0 / np.sqrt(1.0 - speed_squared)
    coordinate_radial_velocity = (
        beta_r / np.sqrt(geometry.base.gamma_rr)
    )
    coordinate_azimuthal_velocity = beta_phi / geometry.radius
    return np.asarray(
        [
            lorentz / geometry.base.lapse,
            lorentz
            * (
                coordinate_radial_velocity
                - geometry.base.radial_shift_over_c
                / geometry.base.lapse
            ),
            lorentz * coordinate_azimuthal_velocity,
        ],
        dtype=float,
    )


def causal_comoving_energy_source(
    geometry: KerrSchildColumnGeometry,
    primitive: ValenciaPerfectFluidPrimitive,
    *,
    comoving_energy_rate: float,
) -> CausalComovingEnergySource:
    """Transform signed comoving column power into conservative sources.

    Positive rate adds energy to the gas. Negative rate removes it. The
    exchange is isotropic in the fluid rest frame, so the four-force is
    parallel to the four-velocity and has no comoving momentum component.
    """

    rate = float(comoving_energy_rate)
    if not np.isfinite(rate):
        raise ValueError("comoving energy rate must be finite")
    four_velocity = _column_four_velocity(geometry, primitive)
    four_force = rate * four_velocity / C**3
    lower_force = geometry.spacetime_metric @ four_force
    alpha = geometry.base.lapse
    killing_source = np.asarray(
        [
            0.0,
            alpha * lower_force[1],
            alpha * lower_force[2],
            -alpha * lower_force[0],
        ],
        dtype=float,
    )
    lower_velocity = geometry.spacetime_metric @ four_velocity
    velocity_force_contraction = float(
        np.dot(lower_velocity, four_force)
    )
    recovered_rate = -C**3 * velocity_force_contraction
    projected_force = (
        four_force + four_velocity * velocity_force_contraction
    )
    scale = max(abs(rate), abs(recovered_rate), 1.0)
    force_scale = max(
        float(np.max(np.abs(four_force))),
        abs(rate) / C**3,
        np.finfo(float).tiny,
    )
    return CausalComovingEnergySource(
        comoving_energy_rate=rate,
        four_force=np.asarray(four_force, dtype=float),
        killing_source_per_ct=killing_source,
        recovered_comoving_rate=float(recovered_rate),
        relative_identity_defect=float(
            abs(recovered_rate - rate) / scale
        ),
        comoving_momentum_relative_defect=float(
            np.max(np.abs(projected_force)) / force_scale
        ),
    )


def causal_diffusion_cooling_rate(
    thermodynamics: GasRadiationColumnThermodynamics,
    *,
    kappa: float = DEFAULT_KAPPA_ES,
    minimum_optical_depth: float = 1.0,
) -> tuple[float, float]:
    """Return positive two-face diffusion cooling and scattering depth."""

    kappa = float(kappa)
    minimum_depth = float(minimum_optical_depth)
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("opacity must be positive and finite")
    if not np.isfinite(minimum_depth) or minimum_depth <= 0.0:
        raise ValueError("minimum optical depth must be positive")
    optical_depth = 0.5 * kappa * thermodynamics.surface_density
    if optical_depth < minimum_depth:
        raise ValueError("diffusion cooling requires an optically thick column")
    cooling = (
        16.0
        * SIGMA_SB
        * thermodynamics.temperature**4
        / (
            3.0
            * kappa
            * thermodynamics.surface_density
        )
    )
    return float(cooling), float(optical_depth)


def causal_thermal_column_source(
    geometry: KerrSchildColumnGeometry,
    eos: QuasiHydrostaticGasRadiationColumnEOS,
    *,
    surface_density: float,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
    temperature: float,
    proper_log_height_rate: float,
    kappa: float = DEFAULT_KAPPA_ES,
) -> CausalThermalColumnSource:
    """Return cooling plus vertical work in the Killing source chart."""

    expansion_rate = float(proper_log_height_rate)
    if not np.isfinite(expansion_rate):
        raise ValueError("proper log-height rate must be finite")
    thermodynamics = eos.from_surface_density_temperature(
        surface_density,
        temperature,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=surface_density,
        radial_velocity_over_c=radial_velocity_over_c,
        azimuthal_velocity_over_c=azimuthal_velocity_over_c,
        specific_internal_energy=thermodynamics.specific_internal_energy,
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    cooling, optical_depth = causal_diffusion_cooling_rate(
        thermodynamics,
        kappa=kappa,
    )
    vertical_work = (
        -thermodynamics.integrated_pressure * expansion_rate
    )
    cooling_source = causal_comoving_energy_source(
        geometry,
        primitive,
        comoving_energy_rate=-cooling,
    )
    vertical_source = causal_comoving_energy_source(
        geometry,
        primitive,
        comoving_energy_rate=vertical_work,
    )
    total = (
        cooling_source.killing_source_per_ct
        + vertical_source.killing_source_per_ct
    )
    return CausalThermalColumnSource(
        thermodynamics=thermodynamics,
        scattering_optical_depth=optical_depth,
        radiative_cooling_rate=cooling,
        vertical_work_rate=float(vertical_work),
        cooling_source=cooling_source,
        vertical_work_source=vertical_source,
        total_killing_source_per_ct=np.asarray(total, dtype=float),
        local_viscous_energy_source=0.0,
    )


def hydrostatic_vertical_work_identity_defect(
    thermodynamics: GasRadiationColumnThermodynamics,
    *,
    surface_density_derivative: float,
    height_derivative: float,
) -> float:
    """Return the enthalpy-column versus 3D pressure-work identity defect."""

    sigma = thermodynamics.surface_density
    height = thermodynamics.proper_half_thickness
    rho = thermodynamics.density
    pi = thermodynamics.integrated_pressure
    pressure = pi / (2.0 * height)
    density_derivative = rho * (
        surface_density_derivative / sigma
        - height_derivative / height
    )
    enthalpy_form = (
        pi * surface_density_derivative / sigma**2
        - pressure * density_derivative / rho**2
    )
    height_form = (
        pressure * height_derivative / (rho * height)
    )
    scale = max(abs(enthalpy_form), abs(height_form), 1.0)
    return float(abs(enthalpy_form - height_form) / scale)


def temporal_vertical_work_per_area(
    old: GasRadiationColumnThermodynamics,
    new: GasRadiationColumnThermodynamics,
) -> float:
    """Return trapezoidal ``Pi dlnH`` correction on the storage side."""

    if (
        old.proper_half_thickness <= 0.0
        or new.proper_half_thickness <= 0.0
    ):
        raise ValueError("column heights must be positive")
    return float(
        0.5
        * (old.integrated_pressure + new.integrated_pressure)
        * np.log(
            new.proper_half_thickness / old.proper_half_thickness
        )
    )


def causal_stress_work_partition(
    *,
    left_angular_velocity: float,
    right_angular_velocity: float,
    left_torque: float,
    right_torque: float,
) -> StressWorkPartition:
    """Split ``Delta(Omega G)`` without adding a total-energy heat source."""

    omega_left = float(left_angular_velocity)
    omega_right = float(right_angular_velocity)
    torque_left = float(left_torque)
    torque_right = float(right_torque)
    values = (omega_left, omega_right, torque_left, torque_right)
    if any(not np.isfinite(value) for value in values):
        raise ValueError("stress-work partition inputs must be finite")
    flux_difference = (
        omega_right * torque_right - omega_left * torque_left
    )
    angular_exchange = (
        0.5
        * (omega_left + omega_right)
        * (torque_right - torque_left)
    )
    shear_conversion = (
        0.5
        * (torque_left + torque_right)
        * (omega_right - omega_left)
    )
    return StressWorkPartition(
        torque_work_flux_difference=float(flux_difference),
        angular_exchange_work=float(angular_exchange),
        shear_conversion_work=float(shear_conversion),
        product_rule_defect=float(
            flux_difference - angular_exchange - shear_conversion
        ),
        explicit_total_energy_heating_source=0.0,
    )
