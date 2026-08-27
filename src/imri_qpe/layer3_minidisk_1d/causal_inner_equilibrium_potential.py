"""Exact fixed-height gas+radiation equilibrium potential current.

For entropy variables ``alpha=mu_mass/T`` and ``beta_mu=u_mu/T``, the
perfect-fluid potential current is

``X^mu = 2 H p(alpha,T) beta^mu``.

Its derivatives generate the surface-mass current and column stress-energy.
This module contains no height dynamics, shear extension, numerical flux, or
trajectory code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import A_RAD, C, DEFAULT_MU_MOL
from imri_qpe.scales import gas_constant_per_gram


@dataclass(frozen=True)
class CompensatedMassAffinity:
    """Two-component representation of the single coordinate ``alpha``.

    This is software extended precision for one severely conditioned scalar,
    not an additional thermodynamic field.  Perturbations of ``alpha`` act on
    the sum of these two components with unit derivative.
    """

    rest_mass_part: float
    thermal_part: float

    @property
    def total(self) -> float:
        return float(self.rest_mass_part + self.thermal_part)


@dataclass(frozen=True)
class EquilibriumColumnPotentialState:
    """Thermodynamics and currents generated at one entropy-variable state."""

    mass_affinity: CompensatedMassAffinity | float
    inverse_temperature_covector: np.ndarray
    temperature: float
    density: float
    pressure: float
    specific_internal_energy: float
    specific_entropy: float
    specific_chemical_potential: float
    four_velocity: np.ndarray
    potential_current: np.ndarray
    surface_mass_current: np.ndarray
    column_stress_energy: np.ndarray
    proper_half_thickness: float


@dataclass(frozen=True)
class EquilibriumColumnPotentialAudit:
    """Thermodynamic and independent derivative defects."""

    density_affinity_roundtrip_relative_defect: float
    chemical_affinity_relative_defect: float
    four_velocity_normalization_defect: float
    first_law_density_relative_defect: float
    first_law_temperature_relative_defect: float
    gibbs_duhem_density_relative_defect: float
    gibbs_duhem_temperature_relative_defect: float
    complex_step_current_jacobian_relative_defect: float
    finite_difference_current_jacobian_relative_defect: float

    @property
    def passed(self) -> bool:
        return (
            self.density_affinity_roundtrip_relative_defect <= 2.0e-9
            and self.chemical_affinity_relative_defect <= 2.0e-15
            and self.four_velocity_normalization_defect <= 2.0e-13
            and self.first_law_density_relative_defect <= 2.0e-14
            and self.first_law_temperature_relative_defect <= 2.0e-14
            and self.gibbs_duhem_density_relative_defect <= 2.0e-14
            and self.gibbs_duhem_temperature_relative_defect <= 2.0e-14
            and self.complex_step_current_jacobian_relative_defect <= 2.0e-9
            and self.finite_difference_current_jacobian_relative_defect <= 2.0e-5
        )


def _require_thermodynamic_inputs(
    density: float,
    temperature: float,
    *,
    gamma_gas: float,
    reference_density: float,
    reference_temperature: float,
) -> tuple[float, float]:
    rho = float(density)
    temp = float(temperature)
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError("density must be positive and finite")
    if not np.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if not np.isfinite(gamma_gas) or gamma_gas <= 1.0:
        raise ValueError("gamma_gas must exceed one")
    if reference_density <= 0.0 or reference_temperature <= 0.0:
        raise ValueError("entropy reference values must be positive")
    return rho, temp


def gas_radiation_specific_chemical_potential(
    density: float,
    temperature: float,
    *,
    mu_mol: float = DEFAULT_MU_MOL,
    gamma_gas: float = 5.0 / 3.0,
    reference_density: float = 1.0,
    reference_temperature: float = 1.0,
) -> float:
    """Return ``mu_mass=c^2+e+p/rho-T*s`` for the shared EOS."""

    rho, temp = _require_thermodynamic_inputs(
        density,
        temperature,
        gamma_gas=gamma_gas,
        reference_density=reference_density,
        reference_temperature=reference_temperature,
    )
    gas_constant = gas_constant_per_gram(mu_mol)
    # In ``e+p/rho-Ts`` the three radiation terms cancel exactly.  Evaluating
    # them separately loses several digits in the low-density states for which
    # this potential is intended, so retain the algebraically reduced form.
    thermal_chemical = gas_constant * temp * (
        gamma_gas / (gamma_gas - 1.0)
        - np.log(temp / reference_temperature) / (gamma_gas - 1.0)
        + np.log(rho / reference_density)
    )
    return float(C**2 + thermal_chemical)


def density_from_mass_affinity(
    mass_affinity: CompensatedMassAffinity | float,
    temperature: float,
    *,
    mu_mol: float = DEFAULT_MU_MOL,
    gamma_gas: float = 5.0 / 3.0,
    reference_density: float = 1.0,
    reference_temperature: float = 1.0,
) -> float:
    """Invert ``alpha=mu_mass/T`` analytically for the density."""

    compensated = isinstance(mass_affinity, CompensatedMassAffinity)
    alpha = (
        float(mass_affinity.total)
        if compensated
        else float(mass_affinity)
    )
    temp = float(temperature)
    if not np.isfinite(alpha) or not np.isfinite(temp) or temp <= 0.0:
        raise ValueError("mass affinity and temperature must be physical")
    # The two leading terms below are O(c^2/(R T)) and nearly cancel.  Carry
    # the scalar inversion in extended precision; returning a float is safe
    # only after the logarithmic density has been recovered.
    gas_constant = float(gas_constant_per_gram(mu_mol))
    if compensated:
        affinity_without_rest_mass = (
            (mass_affinity.rest_mass_part - C**2 / temp)
            + mass_affinity.thermal_part
        )
        log_density_ratio = (
            affinity_without_rest_mass / gas_constant
            - gamma_gas / (gamma_gas - 1.0)
            + np.log(temp / reference_temperature) / (gamma_gas - 1.0)
        )
    else:
        log_density_ratio = (
            alpha / gas_constant
            - C**2 / (gas_constant * temp)
            - gamma_gas / (gamma_gas - 1.0)
            + np.log(temp / reference_temperature) / (gamma_gas - 1.0)
        )
    density = reference_density * np.exp(log_density_ratio)
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("mass affinity maps outside positive density")
    return float(density)


def entropy_variables_from_primitive(
    metric,
    four_velocity,
    *,
    density: float,
    temperature: float,
    mu_mol: float = DEFAULT_MU_MOL,
    gamma_gas: float = 5.0 / 3.0,
    reference_density: float = 1.0,
    reference_temperature: float = 1.0,
) -> tuple[CompensatedMassAffinity, np.ndarray]:
    metric_array = np.asarray(metric, dtype=float)
    velocity = np.asarray(four_velocity, dtype=float)
    if metric_array.shape != (4, 4) or velocity.shape != (4,):
        raise ValueError("metric must be 4x4 and four_velocity length four")
    normalization = velocity @ metric_array @ velocity
    if abs(normalization + 1.0) > 2.0e-12:
        raise ValueError("four_velocity must be unit timelike")
    # Canonicalize the admissible floating-point representative before
    # forming beta.  A one-ulp normalization error in beta is amplified by
    # c^2/(R T) in the density inversion; this normalization changes no
    # physical state but prevents that coordinate artefact from entering the
    # master-potential audit.
    velocity = velocity / np.sqrt(-normalization)
    rho_extended = float(density)
    requested_temperature = float(temperature)
    beta = metric_array @ velocity / requested_temperature
    inverse_metric = np.asarray(
        np.linalg.inv(metric_array), dtype=np.longdouble
    )
    beta_for_inversion = np.asarray(beta, dtype=np.longdouble)
    # The covector is the primary entropy coordinate.  Build alpha from the
    # temperature represented by that exact floating-point beta, which differs
    # from the requested primitive temperature by at most roundoff.
    represented_temperature = 1.0 / np.sqrt(
        -(beta_for_inversion @ (inverse_metric @ beta_for_inversion))
    )
    gas_constant = float(gas_constant_per_gram(mu_mol))
    gamma_extended = float(gamma_gas)
    gamma_minus_one = float(gamma_gas - 1.0)
    thermal_affinity = gas_constant * (
        gamma_extended / gamma_minus_one
        - np.log(
            represented_temperature / reference_temperature
        )
        / gamma_minus_one
        + np.log(rho_extended / reference_density)
    )
    mass_affinity = CompensatedMassAffinity(
        rest_mass_part=C**2 / float(represented_temperature),
        thermal_part=float(thermal_affinity),
    )
    return mass_affinity, beta


def _raw_potential_current(
    inverse_metric: np.ndarray,
    mass_affinity,
    inverse_temperature_covector: np.ndarray,
    proper_half_thickness: float,
    *,
    gas_constant: float,
    gamma_gas: float,
    reference_density: float,
    reference_temperature: float,
    mass_affinity_increment=0.0,
):
    gas_constant = np.longdouble(gas_constant)
    gamma_extended = np.longdouble(gamma_gas)
    gamma_minus_one = np.longdouble(gamma_gas - 1.0)
    reference_temperature_extended = np.longdouble(reference_temperature)
    beta = np.asarray(inverse_temperature_covector)
    beta_contravariant = inverse_metric @ beta
    if np.iscomplexobj(beta):
        # Preserve the exact real-axis evaluation in the severely conditioned
        # alpha/beta chart.  Generic complex sqrt/exp implementations may use
        # a slightly different real-axis algorithm; a one-ulp base difference
        # is then amplified by c^2/(R T).  This decomposition changes only
        # O(h^2) terms and keeps the complex-step derivative independent.
        beta_real = np.asarray(np.real(beta), dtype=np.longdouble)
        beta_imaginary = np.asarray(np.imag(beta), dtype=np.longdouble)
        quadratic_real = (
            beta_real @ inverse_metric @ beta_real
            - beta_imaginary @ inverse_metric @ beta_imaginary
        )
        quadratic_imaginary = 2.0 * (
            beta_real @ inverse_metric @ beta_imaginary
        )
        negative_quadratic = np.clongdouble(
            -quadratic_real - 1j * quadratic_imaginary
        )
        raw_temperature = 1.0 / np.sqrt(negative_quadratic)
        temperature = np.clongdouble(
            1.0 / np.sqrt(-quadratic_real) + 1j * np.imag(raw_temperature)
        )
        logarithmic_temperature = np.clongdouble(
            np.log(
                np.real(temperature) / reference_temperature_extended
            )
            + 1j * np.arctan2(np.imag(temperature), np.real(temperature))
        )
    else:
        beta_squared = beta @ beta_contravariant
        temperature = 1.0 / np.sqrt(-beta_squared)
        logarithmic_temperature = np.log(
            temperature / reference_temperature_extended
        )
    compensated = isinstance(mass_affinity, CompensatedMassAffinity)
    if np.iscomplexobj(temperature) or np.iscomplexobj(mass_affinity_increment):
        reciprocal_temperature_raw = 1.0 / temperature
        reciprocal_temperature = np.clongdouble(
            1.0 / np.real(temperature)
            + 1j * np.imag(reciprocal_temperature_raw)
        )
        rest_coefficient = np.longdouble(C) ** 2
        if compensated:
            real_affinity_without_rest = (
                mass_affinity.rest_mass_part
                - C**2 / float(np.real(temperature))
                + mass_affinity.thermal_part
                + np.real(mass_affinity_increment)
            )
            imaginary_affinity_without_rest = (
                -rest_coefficient * np.imag(reciprocal_temperature)
                + np.imag(mass_affinity_increment)
            )
        else:
            real_affinity_without_rest = (
                np.real(mass_affinity)
                + np.real(mass_affinity_increment)
                - rest_coefficient * np.real(reciprocal_temperature)
            )
            imaginary_affinity_without_rest = (
                np.imag(mass_affinity)
                + np.imag(mass_affinity_increment)
                - rest_coefficient * np.imag(reciprocal_temperature)
            )
        log_density_ratio = np.clongdouble(
            real_affinity_without_rest / gas_constant
            - gamma_extended / gamma_minus_one
            + np.real(logarithmic_temperature) / gamma_minus_one
            + 1j
            * (
                imaginary_affinity_without_rest / gas_constant
                + np.imag(logarithmic_temperature) / gamma_minus_one
            )
        )
    else:
        if compensated:
            affinity_without_rest = (
                mass_affinity.rest_mass_part
                - C**2 / float(temperature)
                + mass_affinity.thermal_part
                + mass_affinity_increment
            )
        else:
            affinity_without_rest = (
                mass_affinity + mass_affinity_increment
                - np.longdouble(C) ** 2 / temperature
            )
        log_density_ratio = (
            affinity_without_rest / gas_constant
            - gamma_extended / gamma_minus_one
            + logarithmic_temperature / gamma_minus_one
        )
    if np.iscomplexobj(log_density_ratio):
        density = reference_density * np.exp(np.real(log_density_ratio)) * (
            np.cos(np.imag(log_density_ratio))
            + 1j * np.sin(np.imag(log_density_ratio))
        )
    else:
        density = reference_density * np.exp(log_density_ratio)
    pressure = (
        density * gas_constant * temperature
        + A_RAD * temperature**4 / 3.0
    )
    return 2.0 * proper_half_thickness * pressure * beta_contravariant


def equilibrium_column_potential_state(
    metric,
    mass_affinity: CompensatedMassAffinity | float,
    inverse_temperature_covector,
    *,
    proper_half_thickness: float,
    mu_mol: float = DEFAULT_MU_MOL,
    gamma_gas: float = 5.0 / 3.0,
    reference_density: float = 1.0,
    reference_temperature: float = 1.0,
) -> EquilibriumColumnPotentialState:
    metric_array = np.asarray(metric, dtype=float)
    beta = np.asarray(inverse_temperature_covector, dtype=np.longdouble)
    height = float(proper_half_thickness)
    if metric_array.shape != (4, 4) or beta.shape != (4,):
        raise ValueError("metric must be 4x4 and beta length four")
    if np.any(~np.isfinite(metric_array)) or np.any(~np.isfinite(beta)):
        raise ValueError("metric and beta must be finite")
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError("proper half thickness must be positive")
    inverse_metric = np.asarray(np.linalg.inv(metric_array), dtype=np.longdouble)
    beta_extended = np.asarray(beta, dtype=np.longdouble)
    beta_contravariant = inverse_metric @ beta_extended
    beta_squared = beta_extended @ beta_contravariant
    if not np.isfinite(beta_squared) or beta_squared >= 0.0:
        raise ValueError("inverse-temperature covector must be timelike")
    temperature_extended = 1.0 / np.sqrt(-beta_squared)
    temperature = float(temperature_extended)
    density = density_from_mass_affinity(
        mass_affinity,
        temperature_extended,
        mu_mol=mu_mol,
        gamma_gas=gamma_gas,
        reference_density=reference_density,
        reference_temperature=reference_temperature,
    )
    gas_constant = np.longdouble(gas_constant_per_gram(mu_mol))
    density_extended = np.longdouble(density)
    pressure = (
        density_extended * gas_constant * temperature_extended
        + np.longdouble(A_RAD) * temperature_extended**4 / 3.0
    )
    internal = (
        gas_constant * temperature_extended / np.longdouble(gamma_gas - 1.0)
        + np.longdouble(A_RAD) * temperature_extended**4 / density_extended
    )
    entropy = (
        gas_constant / np.longdouble(gamma_gas - 1.0)
        * np.log(temperature_extended / np.longdouble(reference_temperature))
        - gas_constant
        * np.log(density_extended / np.longdouble(reference_density))
        + 4.0
        * np.longdouble(A_RAD)
        * temperature_extended**3
        / (3.0 * density_extended)
    )
    chemical = gas_radiation_specific_chemical_potential(
        density,
        temperature,
        mu_mol=mu_mol,
        gamma_gas=gamma_gas,
        reference_density=reference_density,
        reference_temperature=reference_temperature,
    )
    velocity = temperature_extended * beta_contravariant
    energy_density = density_extended * (np.longdouble(C) ** 2 + internal)
    potential = 2.0 * height * pressure * beta_contravariant
    mass_current = 2.0 * height * density_extended * velocity
    stress_energy = 2.0 * height * (
        (energy_density + pressure) * np.outer(velocity, velocity)
        + pressure * inverse_metric
    )
    return EquilibriumColumnPotentialState(
        mass_affinity=mass_affinity,
        inverse_temperature_covector=beta.copy(),
        temperature=temperature,
        density=density,
        pressure=float(pressure),
        specific_internal_energy=float(internal),
        specific_entropy=float(entropy),
        specific_chemical_potential=float(chemical),
        four_velocity=velocity,
        potential_current=potential,
        surface_mass_current=mass_current,
        column_stress_energy=stress_energy,
        proper_half_thickness=height,
    )


def analytic_potential_current_jacobian(
    state: EquilibriumColumnPotentialState,
) -> np.ndarray:
    """Return columns ``dX/d(alpha,beta_0,...,beta_3)``."""

    return np.column_stack((state.surface_mass_current, state.column_stress_energy))


def complex_step_potential_current_jacobian(
    metric,
    mass_affinity: CompensatedMassAffinity | float,
    inverse_temperature_covector,
    *,
    proper_half_thickness: float,
    mu_mol: float = DEFAULT_MU_MOL,
    gamma_gas: float = 5.0 / 3.0,
    reference_density: float = 1.0,
    reference_temperature: float = 1.0,
) -> np.ndarray:
    metric_array = np.asarray(metric, dtype=float)
    inverse_metric = np.asarray(np.linalg.inv(metric_array), dtype=np.longdouble)
    beta = np.asarray(inverse_temperature_covector, dtype=np.longdouble)
    gas_constant = gas_constant_per_gram(mu_mol)
    coordinates = np.concatenate(([0.0], beta)).astype(np.longdouble)
    steps = np.concatenate(
        ((1.0e-20 * gas_constant,), 1.0e-20 * np.maximum(np.abs(beta), 1.0))
    )
    jacobian = np.empty((4, 5), dtype=float)
    for index, step in enumerate(steps):
        perturbed = coordinates.astype(np.clongdouble)
        perturbed[index] += 1j * step
        value = _raw_potential_current(
            inverse_metric,
            mass_affinity,
            perturbed[1:],
            proper_half_thickness,
            gas_constant=gas_constant,
            gamma_gas=gamma_gas,
            reference_density=reference_density,
            reference_temperature=reference_temperature,
            mass_affinity_increment=perturbed[0],
        )
        jacobian[:, index] = np.imag(value) / step
    return jacobian


def finite_difference_potential_current_jacobian(
    metric,
    mass_affinity: CompensatedMassAffinity | float,
    inverse_temperature_covector,
    *,
    proper_half_thickness: float,
    mu_mol: float = DEFAULT_MU_MOL,
    gamma_gas: float = 5.0 / 3.0,
    reference_density: float = 1.0,
    reference_temperature: float = 1.0,
    step_factor: float = 0.5,
) -> np.ndarray:
    metric_array = np.asarray(metric, dtype=float)
    inverse_metric = np.asarray(np.linalg.inv(metric_array), dtype=np.longdouble)
    beta = np.asarray(inverse_temperature_covector, dtype=np.longdouble)
    gas_constant = gas_constant_per_gram(mu_mol)
    coordinates = np.concatenate(([0.0], beta)).astype(np.longdouble)
    temperature = float(1.0 / np.sqrt(-(beta @ inverse_metric @ beta)))
    rest_mass_condition = C**2 / (gas_constant * temperature)
    # At fixed alpha a relative beta perturbation is amplified by
    # c^2/(R T) in log(rho).  Scale it so every stencil point remains in the
    # same thermodynamic neighbourhood instead of crossing density e-folds.
    inverse_diagonal = np.abs(np.diag(inverse_metric))
    metric_covector_scale = 1.0 / (
        np.longdouble(temperature)
        * np.sqrt(np.maximum(inverse_diagonal, np.finfo(float).tiny))
    )
    beta_steps = 2.0e-4 * np.maximum(
        np.abs(beta), 0.25 * metric_covector_scale
    ) / rest_mass_condition
    # A 0.06 logarithmic-density scale is large enough to dominate
    # subtraction noise in the rest-mass-conditioned chart while the
    # sixth-order stencil keeps truncation error small.
    steps = step_factor * np.asarray(
        np.concatenate(((6.0e-2 * gas_constant,), 300.0 * beta_steps)),
        dtype=np.longdouble,
    )

    def evaluate(point):
        return np.asarray(
            _raw_potential_current(
                inverse_metric,
                mass_affinity,
                point[1:],
                proper_half_thickness,
                gas_constant=gas_constant,
                gamma_gas=gamma_gas,
                reference_density=reference_density,
                reference_temperature=reference_temperature,
                mass_affinity_increment=point[0],
            ),
            dtype=np.longdouble,
        )

    jacobian = np.empty((4, 5), dtype=float)
    for index, step in enumerate(steps):
        direction = np.zeros(5, dtype=np.longdouble)
        direction[index] = step
        jacobian[:, index] = (
            -evaluate(coordinates - 3.0 * direction)
            + 9.0 * evaluate(coordinates - 2.0 * direction)
            - 45.0 * evaluate(coordinates - direction)
            + 45.0 * evaluate(coordinates + direction)
            - 9.0 * evaluate(coordinates + 2.0 * direction)
            + evaluate(coordinates + 3.0 * direction)
        ) / (60.0 * step)
    return jacobian


def _columnwise_relative_defect(actual: np.ndarray, expected: np.ndarray) -> float:
    scales = np.maximum(np.max(np.abs(expected), axis=0), np.finfo(float).tiny)
    return float(np.max(np.abs(actual - expected) / scales[None, :]))


def audit_equilibrium_column_potential(
    metric,
    four_velocity,
    *,
    density: float,
    temperature: float,
    proper_half_thickness: float,
    mu_mol: float = DEFAULT_MU_MOL,
    gamma_gas: float = 5.0 / 3.0,
) -> EquilibriumColumnPotentialAudit:
    metric_array = np.asarray(metric, dtype=float)
    velocity = np.asarray(four_velocity, dtype=float)
    alpha, beta = entropy_variables_from_primitive(
        metric_array,
        velocity,
        density=density,
        temperature=temperature,
        mu_mol=mu_mol,
        gamma_gas=gamma_gas,
    )
    state = equilibrium_column_potential_state(
        metric_array,
        alpha,
        beta,
        proper_half_thickness=proper_half_thickness,
        mu_mol=mu_mol,
        gamma_gas=gamma_gas,
    )
    analytic = analytic_potential_current_jacobian(state)
    complex_jacobian = complex_step_potential_current_jacobian(
        metric_array,
        alpha,
        beta,
        proper_half_thickness=proper_half_thickness,
        mu_mol=mu_mol,
        gamma_gas=gamma_gas,
    )
    finite_jacobian = finite_difference_potential_current_jacobian(
        metric_array,
        alpha,
        beta,
        proper_half_thickness=proper_half_thickness,
        mu_mol=mu_mol,
        gamma_gas=gamma_gas,
    )
    rho = float(density)
    temp = float(temperature)
    gas_constant = gas_constant_per_gram(mu_mol)
    pressure = rho * gas_constant * temp + A_RAD * temp**4 / 3.0
    de_drho = -A_RAD * temp**4 / rho**2
    de_dtemp = gas_constant / (gamma_gas - 1.0) + 4.0 * A_RAD * temp**3 / rho
    ds_drho = -gas_constant / rho - 4.0 * A_RAD * temp**3 / (3.0 * rho**2)
    ds_dtemp = gas_constant / ((gamma_gas - 1.0) * temp) + 4.0 * A_RAD * temp**2 / rho
    dmu_drho = gas_constant * temp / rho
    dmu_dtemp = (
        gas_constant * gamma_gas / (gamma_gas - 1.0)
        - gas_constant * np.log(temp) / (gamma_gas - 1.0)
        + gas_constant * np.log(rho)
        - gas_constant / (gamma_gas - 1.0)
    )
    entropy = (
        gas_constant / (gamma_gas - 1.0) * np.log(temp)
        - gas_constant * np.log(rho)
        + 4.0 * A_RAD * temp**3 / (3.0 * rho)
    )
    entropy_volume = rho * entropy
    dp_drho = gas_constant * temp
    dp_dtemp = rho * gas_constant + 4.0 * A_RAD * temp**3 / 3.0
    density_first = de_drho - temp * ds_drho - pressure / rho**2
    temperature_first = de_dtemp - temp * ds_dtemp
    density_gibbs = dp_drho - rho * dmu_drho
    temperature_gibbs = dp_dtemp - (entropy_volume + rho * dmu_dtemp)
    alpha_total = (
        alpha.total if isinstance(alpha, CompensatedMassAffinity) else float(alpha)
    )
    return EquilibriumColumnPotentialAudit(
        density_affinity_roundtrip_relative_defect=abs(state.density - rho) / rho,
        chemical_affinity_relative_defect=abs(state.specific_chemical_potential / temp - alpha_total) / max(abs(alpha_total), 1.0),
        four_velocity_normalization_defect=abs(float(state.four_velocity @ metric_array @ state.four_velocity) + 1.0),
        first_law_density_relative_defect=abs(density_first) / max(abs(de_drho), abs(temp * ds_drho), abs(pressure / rho**2), 1.0),
        first_law_temperature_relative_defect=abs(temperature_first) / max(abs(de_dtemp), abs(temp * ds_dtemp), 1.0),
        gibbs_duhem_density_relative_defect=abs(density_gibbs) / max(abs(dp_drho), abs(rho * dmu_drho), 1.0),
        gibbs_duhem_temperature_relative_defect=abs(temperature_gibbs) / max(abs(dp_dtemp), abs(entropy_volume + rho * dmu_dtemp), 1.0),
        complex_step_current_jacobian_relative_defect=_columnwise_relative_defect(complex_jacobian, analytic),
        finite_difference_current_jacobian_relative_defect=_columnwise_relative_defect(finite_jacobian, analytic),
    )


__all__ = [
    "CompensatedMassAffinity",
    "EquilibriumColumnPotentialAudit",
    "EquilibriumColumnPotentialState",
    "analytic_potential_current_jacobian",
    "audit_equilibrium_column_potential",
    "complex_step_potential_current_jacobian",
    "density_from_mass_affinity",
    "entropy_variables_from_primitive",
    "equilibrium_column_potential_state",
    "finite_difference_potential_current_jacobian",
    "gas_radiation_specific_chemical_potential",
]
