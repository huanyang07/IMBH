"""Gas+radiation primitive recovery for the causal Valencia column chart."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from imri_qpe.constants import A_RAD, C, DEFAULT_MU_MOL
from imri_qpe.scales import gas_constant_per_gram

from .causal_inner import (
    gas_radiation_relativistic_sound_speed_squared,
)
from .causal_inner_valencia import (
    SchwarzschildKerrSchildGeometry,
    ValenciaCharacteristicAudit,
    ValenciaColumnState,
    valencia_column_state,
    valencia_radial_characteristic_speeds_over_c,
)


@dataclass(frozen=True)
class GasRadiationColumnThermodynamics:
    """One fixed-height gas+radiation column state."""

    surface_density: float
    temperature: float
    proper_half_thickness: float
    density: float
    integrated_pressure: float
    specific_internal_energy: float
    specific_enthalpy: float
    sound_speed: float


@dataclass(frozen=True)
class FixedHeightGasRadiationColumnEOS:
    """Gravity-independent column EOS used to certify primitive recovery."""

    proper_half_thickness: float
    mu_mol: float = DEFAULT_MU_MOL
    gamma_gas: float = 5.0 / 3.0
    minimum_temperature: float = 1.0e-3
    maximum_temperature: float = 1.0e13

    def __post_init__(self) -> None:
        values = (
            self.proper_half_thickness,
            self.mu_mol,
            self.minimum_temperature,
            self.maximum_temperature,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("column EOS parameters must be finite")
        if self.proper_half_thickness <= 0.0 or self.mu_mol <= 0.0:
            raise ValueError("column height and molecular weight must be positive")
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
        """Return the exact shared EOS at a fixed proper column height."""

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
        height = float(self.proper_half_thickness)
        density = sigma / (2.0 * height)
        gas_constant = gas_constant_per_gram(self.mu_mol)
        gas_pressure = density * gas_constant * temperature
        radiation_pressure = A_RAD * temperature**4 / 3.0
        integrated_pressure = 2.0 * height * (
            gas_pressure + radiation_pressure
        )
        internal_energy = (
            gas_constant * temperature / (self.gamma_gas - 1.0)
            + A_RAD * temperature**4 / density
        )
        specific_enthalpy = internal_energy + integrated_pressure / sigma
        sound_speed = np.sqrt(
            gas_radiation_relativistic_sound_speed_squared(
                density,
                temperature,
                mu_mol=self.mu_mol,
                gamma_gas=self.gamma_gas,
            )
        )
        return GasRadiationColumnThermodynamics(
            surface_density=sigma,
            temperature=temperature,
            proper_half_thickness=height,
            density=float(density),
            integrated_pressure=float(integrated_pressure),
            specific_internal_energy=float(internal_energy),
            specific_enthalpy=float(specific_enthalpy),
            sound_speed=float(sound_speed),
        )

    def from_surface_density_internal_energy(
        self,
        surface_density: float,
        specific_internal_energy: float,
    ) -> GasRadiationColumnThermodynamics:
        """Invert the monotone gas+radiation internal energy for temperature."""

        sigma = float(surface_density)
        target = float(specific_internal_energy)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("surface density must be positive and finite")
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("specific internal energy must be positive and finite")

        density = sigma / (2.0 * self.proper_half_thickness)
        gas_coefficient = (
            gas_constant_per_gram(self.mu_mol) / (self.gamma_gas - 1.0)
        )

        def residual(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            return (
                gas_coefficient * temperature
                + A_RAD * temperature**4 / density
                - target
            )

        lower = float(np.log(self.minimum_temperature))
        upper = float(np.log(self.maximum_temperature))
        lower_residual = residual(lower)
        upper_residual = residual(upper)
        if lower_residual > 0.0 or upper_residual < 0.0:
            raise ValueError("internal energy lies outside the column EOS bounds")
        root = brentq(
            residual,
            lower,
            upper,
            xtol=1.0e-13,
            rtol=1.0e-13,
            maxiter=160,
        )
        return self.from_surface_density_temperature(sigma, float(np.exp(root)))


@dataclass(frozen=True)
class ValenciaGasRadiationPrimitive:
    """Recovered primitive chart and thermodynamic column."""

    surface_density: float
    radial_velocity_over_c: float
    azimuthal_velocity_over_c: float
    temperature: float
    thermodynamics: GasRadiationColumnThermodynamics


@dataclass(frozen=True)
class ValenciaPrimitiveRecoveryAudit:
    """One pressure-root primitive recovery and round-trip audit."""

    primitive: ValenciaGasRadiationPrimitive
    state: ValenciaColumnState
    pressure_root_iterations: int
    pressure_root_function_calls: int
    maximum_relative_conserved_defect: float


def valencia_gas_radiation_column_state(
    geometry: SchwarzschildKerrSchildGeometry,
    eos: FixedHeightGasRadiationColumnEOS,
    *,
    surface_density: float,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
    temperature: float,
) -> tuple[ValenciaColumnState, GasRadiationColumnThermodynamics]:
    """Map the gas+radiation primitive chart to conservative variables."""

    thermodynamics = eos.from_surface_density_temperature(
        surface_density,
        temperature,
    )
    state = valencia_column_state(
        geometry,
        surface_density=surface_density,
        radial_velocity_over_c=radial_velocity_over_c,
        azimuthal_velocity_over_c=azimuthal_velocity_over_c,
        specific_internal_energy=thermodynamics.specific_internal_energy,
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    return state, thermodynamics


def recover_valencia_gas_radiation_primitives(
    geometry: SchwarzschildKerrSchildGeometry,
    eos: FixedHeightGasRadiationColumnEOS,
    conserved,
    *,
    maximum_pressure_ratio: float = 1.0e4,
) -> ValenciaPrimitiveRecoveryAudit:
    """Recover ``(Sigma,v_R,v_phi,T)`` from one Valencia column state."""

    target = np.asarray(conserved, dtype=float)
    if target.shape != (4,) or np.any(~np.isfinite(target)):
        raise ValueError("Valencia conserved state must be a finite length-four vector")
    rest_mass, radial_momentum, angular_momentum, energy = map(float, target)
    if rest_mass <= 0.0:
        raise ValueError("Valencia rest-mass density must be positive")
    if not np.isfinite(maximum_pressure_ratio) or maximum_pressure_ratio <= 1.0:
        raise ValueError("maximum pressure ratio must exceed one")

    extended = np.longdouble
    momentum_squared_extended = (
        extended(geometry.inverse_gamma_rr) * extended(radial_momentum) ** 2
        + extended(angular_momentum) ** 2
        / extended(geometry.gamma_phiphi)
    )
    momentum_squared = float(momentum_squared_extended)
    momentum_magnitude = float(np.sqrt(max(momentum_squared, 0.0)))
    scale = max(abs(energy) + rest_mass + momentum_magnitude, rest_mass)
    minimum_pressure = max(
        momentum_magnitude - energy - rest_mass,
        0.0,
    )
    minimum_ratio = max(
        minimum_pressure / scale * (1.0 + 1.0e-12),
        1.0e-14,
    )
    if minimum_ratio >= maximum_pressure_ratio:
        raise ValueError("could not bracket the Valencia pressure root")
    evaluations = 0

    def candidate(pressure_ratio: float):
        nonlocal evaluations
        evaluations += 1
        pressure_mass_extended = extended(pressure_ratio) * extended(scale)
        total_enthalpy_density_extended = (
            extended(energy)
            + extended(rest_mass)
            + pressure_mass_extended
        )
        if total_enthalpy_density_extended <= extended(momentum_magnitude):
            raise ValueError("trial pressure gives a non-timelike state")
        timelike_denominator_extended = (
            total_enthalpy_density_extended**2
            - momentum_squared_extended
        )
        lorentz_extended = total_enthalpy_density_extended / np.sqrt(
            timelike_denominator_extended
        )
        sigma_extended = extended(rest_mass) / lorentz_extended
        lorentz_squared_minus_one_extended = (
            momentum_squared_extended / timelike_denominator_extended
        )
        lorentz_minus_one_extended = (
            lorentz_squared_minus_one_extended / (lorentz_extended + 1.0)
        )
        internal_over_c2_extended = (
            extended(energy)
            - extended(rest_mass) * lorentz_minus_one_extended
            - pressure_mass_extended
            * lorentz_squared_minus_one_extended
        ) / (
            extended(rest_mass) * lorentz_extended
        )
        internal_over_c2 = float(internal_over_c2_extended)
        if (
            not np.isfinite(internal_over_c2)
            or internal_over_c2 <= 0.0
        ):
            raise ValueError("trial pressure gives non-positive internal energy")
        sigma = float(sigma_extended)
        thermodynamics = eos.from_surface_density_internal_energy(
            sigma,
            internal_over_c2 * C**2,
        )
        residual = (
            float(pressure_mass_extended)
            - thermodynamics.integrated_pressure / C**2
        ) / scale
        return float(residual), thermodynamics, float(lorentz_extended)

    bracket = None
    previous = None
    ratios = np.geomspace(
        minimum_ratio,
        float(maximum_pressure_ratio),
        240,
    )
    for ratio in ratios:
        try:
            residual, _thermodynamics, _lorentz = candidate(float(ratio))
        except ValueError:
            continue
        if residual == 0.0:
            bracket = (float(ratio), float(ratio))
            break
        if previous is not None and np.sign(previous[1]) != np.sign(residual):
            bracket = (previous[0], float(ratio))
            break
        previous = (float(ratio), residual)
    if bracket is None:
        raise ValueError("could not bracket the Valencia pressure root")

    if bracket[0] == bracket[1]:
        pressure_ratio = bracket[0]
        iterations = 0
    else:
        root, root_result = brentq(
            lambda value: candidate(float(np.exp(value)))[0],
            float(np.log(bracket[0])),
            float(np.log(bracket[1])),
            xtol=1.0e-13,
            rtol=1.0e-13,
            maxiter=160,
            full_output=True,
        )
        if not root_result.converged:
            raise RuntimeError("Valencia pressure root did not converge")
        pressure_ratio = float(np.exp(root))
        iterations = int(root_result.iterations)

    pressure_mass_extended = extended(pressure_ratio) * extended(scale)
    total_enthalpy_density_extended = (
        extended(energy)
        + extended(rest_mass)
        + pressure_mass_extended
    )
    lorentz_extended = total_enthalpy_density_extended / np.sqrt(
        total_enthalpy_density_extended**2 - momentum_squared_extended
    )
    sigma = float(extended(rest_mass) / lorentz_extended)
    beta_r = float(
        extended(radial_momentum)
        / (
            total_enthalpy_density_extended
            * np.sqrt(extended(geometry.gamma_rr))
        )
    )
    beta_phi = float(
        extended(angular_momentum)
        / (
            total_enthalpy_density_extended * extended(geometry.radius)
        )
    )
    residual, thermodynamics, _ = candidate(pressure_ratio)
    if abs(residual) > 2.0e-11:
        raise RuntimeError("Valencia pressure root fails its residual gate")
    recovered_state, recovered_thermodynamics = (
        valencia_gas_radiation_column_state(
            geometry,
            eos,
            surface_density=sigma,
            radial_velocity_over_c=beta_r,
            azimuthal_velocity_over_c=beta_phi,
            temperature=thermodynamics.temperature,
        )
    )
    denominator = np.maximum(np.abs(target), rest_mass * 1.0e-12)
    defect = float(
        np.max(np.abs(recovered_state.conserved - target) / denominator)
    )
    primitive = ValenciaGasRadiationPrimitive(
        surface_density=float(sigma),
        radial_velocity_over_c=float(beta_r),
        azimuthal_velocity_over_c=float(beta_phi),
        temperature=float(recovered_thermodynamics.temperature),
        thermodynamics=recovered_thermodynamics,
    )
    return ValenciaPrimitiveRecoveryAudit(
        primitive=primitive,
        state=recovered_state,
        pressure_root_iterations=iterations,
        pressure_root_function_calls=evaluations,
        maximum_relative_conserved_defect=defect,
    )


def audit_gas_radiation_valencia_eigensystem(
    geometry: SchwarzschildKerrSchildGeometry,
    eos: FixedHeightGasRadiationColumnEOS,
    *,
    surface_density: float,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
    temperature: float,
    finite_difference_step: float = 1.0e-3,
) -> ValenciaCharacteristicAudit:
    """Compare the gas+radiation flux Jacobian with Valencia characteristics."""

    if not 0.0 < finite_difference_step < 1.0e-2:
        raise ValueError("finite-difference step must be positive and small")
    chart = np.asarray(
        [
            np.log(surface_density),
            radial_velocity_over_c,
            azimuthal_velocity_over_c,
            np.log(temperature),
        ],
        dtype=float,
    )

    def state(values: np.ndarray):
        result, thermodynamics = valencia_gas_radiation_column_state(
            geometry,
            eos,
            surface_density=float(np.exp(values[0])),
            radial_velocity_over_c=float(values[1]),
            azimuthal_velocity_over_c=float(values[2]),
            temperature=float(np.exp(values[3])),
        )
        return result, thermodynamics

    conserved_jacobian = np.empty((4, 4), dtype=float)
    flux_jacobian = np.empty((4, 4), dtype=float)
    for index in range(4):
        plus = np.array(chart, copy=True)
        minus = np.array(chart, copy=True)
        plus_two = np.array(chart, copy=True)
        minus_two = np.array(chart, copy=True)
        plus[index] += finite_difference_step
        minus[index] -= finite_difference_step
        plus_two[index] += 2.0 * finite_difference_step
        minus_two[index] -= 2.0 * finite_difference_step
        plus_state, _ = state(plus)
        minus_state, _ = state(minus)
        plus_two_state, _ = state(plus_two)
        minus_two_state, _ = state(minus_two)
        conserved_jacobian[:, index] = (
            minus_two_state.conserved
            - 8.0 * minus_state.conserved
            + 8.0 * plus_state.conserved
            - plus_two_state.conserved
        ) / (12.0 * finite_difference_step)
        flux_jacobian[:, index] = (
            minus_two_state.flux_over_c
            - 8.0 * minus_state.flux_over_c
            + 8.0 * plus_state.flux_over_c
            - plus_two_state.flux_over_c
        ) / (12.0 * finite_difference_step)

    conservative_jacobian = np.linalg.solve(
        conserved_jacobian.T,
        flux_jacobian.T,
    ).T
    numerical_values = np.linalg.eigvals(conservative_jacobian)
    if np.max(np.abs(np.imag(numerical_values))) > 1.0e-8:
        raise ValueError("gas+radiation Valencia eigenvalues are not real")
    numerical = np.sort(np.real(numerical_values))
    reference, thermodynamics = state(chart)
    analytic = np.sort(
        np.asarray(
            valencia_radial_characteristic_speeds_over_c(
                geometry,
                radial_velocity_over_c=radial_velocity_over_c,
                azimuthal_velocity_over_c=azimuthal_velocity_over_c,
                sound_speed_over_c=thermodynamics.sound_speed / C,
            ),
            dtype=float,
        )
    )
    component_scale = np.maximum(
        np.abs(reference.conserved),
        np.max(np.abs(reference.conserved)) * 1.0e-12,
    )
    scaled_jacobian = (
        conservative_jacobian
        * component_scale[np.newaxis, :]
        / component_scale[:, np.newaxis]
    )
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    rank_threshold = max(
        float(np.max(np.abs(numerical))) * 1.0e-8,
        1.0e-10,
    )
    return ValenciaCharacteristicAudit(
        analytic_speeds_over_c=tuple(float(value) for value in analytic),
        numerical_speeds_over_c=tuple(float(value) for value in numerical),
        incoming_inner_characteristics=int(np.sum(analytic > 0.0)),
        stationary_flux_rank=int(
            np.sum(np.abs(numerical) > rank_threshold)
        ),
        smallest_stationary_singular_value=float(singular_values[-1]),
        maximum_eigenvalue_defect=float(np.max(np.abs(analytic - numerical))),
    )
