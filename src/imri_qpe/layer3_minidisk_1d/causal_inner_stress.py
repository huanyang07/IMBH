"""Causal relativistic alpha stress for the Kerr-Schild column."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C

from .causal_inner_geometry import (
    KerrSchildColumnGeometry,
    ValenciaPerfectFluidPrimitive,
    audit_kerr_schild_column_sources,
)
from .causal_inner_recovery import FixedHeightGasRadiationColumnEOS
from .causal_inner_valencia import (
    valencia_radial_characteristic_speeds_over_c,
)


@dataclass(frozen=True)
class CausalAlphaShearClosure:
    """Local Maxwell-Cattaneo calibration of the common alpha stress.

    ``specific_shear_viscosity_seconds`` multiplies the positive rest-frame
    shear rate to give the dimensionless stress ``W/(Sigma c^2)``. The
    relaxation time is selected so the transverse signal speed is finite:

    ``c_nu^2/c^2 = nu_s / (tau h)``.
    """

    equilibrium_specific_stress: float
    reference_positive_shear_rate: float
    specific_shear_viscosity_seconds: float
    relaxation_time: float
    viscous_signal_speed_over_c: float
    specific_enthalpy_over_c2: float

    def target_specific_stress(
        self,
        positive_shear_rate: float,
    ) -> float:
        """Return the Navier-Stokes target for a resolved shear rate."""

        shear_rate = float(positive_shear_rate)
        if not np.isfinite(shear_rate):
            raise ValueError("positive shear rate must be finite")
        return float(self.specific_shear_viscosity_seconds * shear_rate)


@dataclass(frozen=True)
class CausalStressColumnState:
    """Killing-chart state and flux for one rest-frame shear stress."""

    killing_conserved: np.ndarray
    killing_flux_over_c: np.ndarray
    stress_killing_conserved_increment: np.ndarray
    stress_killing_flux_increment_over_c: np.ndarray
    viscous_stress_tensor: np.ndarray
    relaxing_stress_conserved: float
    relaxing_stress_flux_over_c: float
    specific_stress: float
    lorentz_factor: float
    coordinate_angular_velocity: float
    radial_geometric_source_increment: float
    tensor_trace_relative_defect: float
    tensor_orthogonality_relative_defect: float
    radial_work_relative_defect: float


@dataclass(frozen=True)
class CausalStressCharacteristicAudit:
    """Frozen-geometry characteristic audit of the causal shear system."""

    speeds_over_c: tuple[float, ...]
    acoustic_speeds_over_c: tuple[float, float]
    shear_speeds_over_c: tuple[float, float]
    local_rest_shear_speeds_over_c: tuple[float, float]
    incoming_inner_characteristics: int
    stationary_flux_rank: int
    smallest_absolute_characteristic_speed: float
    shear_principal_eigenvalue_defect: float
    maximum_imaginary_eigenvalue: float
    maximum_light_cone_excess: float

    @property
    def causal_and_hyperbolic(self) -> bool:
        return (
            self.maximum_imaginary_eigenvalue <= 1.0e-13
            and self.shear_principal_eigenvalue_defect <= 1.0e-12
            and self.maximum_light_cone_excess <= 1.0e-12
        )

    @property
    def causally_outgoing_inner_edge(self) -> bool:
        return self.incoming_inner_characteristics == 0


@dataclass(frozen=True)
class AdvectedStressFluxAudit:
    """Rejected control with stress advection but no shear principal term."""

    eigenvalues: tuple[complex, ...]
    maximum_imaginary_eigenvalue: float
    maximum_light_cone_excess: float

    @property
    def hyperbolic(self) -> bool:
        return self.maximum_imaginary_eigenvalue <= 1.0e-10


def equilibrium_alpha_specific_stress(
    primitive: ValenciaPerfectFluidPrimitive,
    *,
    alpha: float,
    stress_factor: float = 1.0,
) -> float:
    """Return the positive alpha-stress magnitude divided by ``Sigma c^2``."""

    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be non-negative and finite")
    if not np.isfinite(stress_factor) or stress_factor <= 0.0:
        raise ValueError("stress_factor must be positive and finite")
    sigma = float(primitive.surface_density)
    pressure = float(primitive.integrated_pressure)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("surface density must be positive and finite")
    if not np.isfinite(pressure) or pressure < 0.0:
        raise ValueError("integrated pressure must be finite and non-negative")
    return float(stress_factor * alpha * pressure / (sigma * C**2))


def calibrate_causal_alpha_shear(
    primitive: ValenciaPerfectFluidPrimitive,
    *,
    alpha: float,
    reference_positive_shear_rate: float,
    viscous_signal_speed_over_c: float,
    stress_factor: float = 1.0,
) -> CausalAlphaShearClosure:
    """Calibrate a finite-speed shear law to the common alpha stress.

    The alpha prescription fixes the equilibrium amplitude only at the
    supplied reference shear. The resolved Navier-Stokes target remains
    proportional to the actual shear, which supplies the spatial principal
    coupling required by a telegraph-type causal closure.
    """

    shear_rate = float(reference_positive_shear_rate)
    signal_speed = float(viscous_signal_speed_over_c)
    if not np.isfinite(shear_rate) or shear_rate <= 0.0:
        raise ValueError("reference positive shear rate must be positive")
    if not np.isfinite(signal_speed) or not 0.0 < signal_speed < 1.0:
        raise ValueError("viscous signal speed must lie strictly in (0,c)")
    equilibrium = equilibrium_alpha_specific_stress(
        primitive,
        alpha=alpha,
        stress_factor=stress_factor,
    )
    if equilibrium <= 0.0:
        raise ValueError("causal alpha calibration requires positive stress")
    sigma = float(primitive.surface_density)
    enthalpy = (
        1.0
        + float(primitive.specific_internal_energy) / C**2
        + float(primitive.integrated_pressure) / (sigma * C**2)
    )
    specific_viscosity = equilibrium / shear_rate
    relaxation_time = (
        specific_viscosity / (enthalpy * signal_speed**2)
    )
    return CausalAlphaShearClosure(
        equilibrium_specific_stress=float(equilibrium),
        reference_positive_shear_rate=shear_rate,
        specific_shear_viscosity_seconds=float(specific_viscosity),
        relaxation_time=float(relaxation_time),
        viscous_signal_speed_over_c=signal_speed,
        specific_enthalpy_over_c2=float(enthalpy),
    )


def _fluid_rest_tetrad(
    geometry: KerrSchildColumnGeometry,
    primitive: ValenciaPerfectFluidPrimitive,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(u, e_R, e_phi)`` as contravariant coordinate vectors."""

    beta_r = float(primitive.radial_velocity_over_c)
    beta_phi = float(primitive.azimuthal_velocity_over_c)
    speed_squared = beta_r**2 + beta_phi**2
    if speed_squared >= 1.0:
        raise ValueError("Eulerian three-velocity must be subluminal")

    normal = np.asarray(
        [
            1.0 / geometry.base.lapse,
            -geometry.base.radial_shift_over_c / geometry.base.lapse,
            0.0,
        ],
        dtype=float,
    )
    radial = np.asarray(
        [0.0, 1.0 / np.sqrt(geometry.base.gamma_rr), 0.0],
        dtype=float,
    )
    azimuthal = np.asarray(
        [0.0, 0.0, 1.0 / geometry.radius],
        dtype=float,
    )
    gamma_phi = 1.0 / np.sqrt(1.0 - beta_phi**2)
    corotating_time = gamma_phi * (normal + beta_phi * azimuthal)
    corotating_phi = gamma_phi * (beta_phi * normal + azimuthal)
    corotating_radial_speed = gamma_phi * beta_r
    gamma_radial = 1.0 / np.sqrt(1.0 - corotating_radial_speed**2)
    four_velocity = gamma_radial * (
        corotating_time + corotating_radial_speed * radial
    )
    rest_radial = gamma_radial * (
        corotating_radial_speed * corotating_time + radial
    )
    return four_velocity, rest_radial, corotating_phi


def causal_stress_column_state(
    geometry: KerrSchildColumnGeometry,
    primitive: ValenciaPerfectFluidPrimitive,
    *,
    specific_stress: float,
) -> CausalStressColumnState:
    """Transform one rest-frame ``R-phi`` stress into the Killing chart."""

    specific_stress = float(specific_stress)
    if not np.isfinite(specific_stress):
        raise ValueError("specific stress must be finite")
    sigma = float(primitive.surface_density)
    specific_enthalpy = (
        1.0
        + float(primitive.specific_internal_energy) / C**2
        + float(primitive.integrated_pressure) / (sigma * C**2)
    )
    if abs(specific_stress) >= specific_enthalpy:
        raise ValueError("shear stress violates the column dominant-energy gate")

    perfect = audit_kerr_schild_column_sources(geometry, primitive)
    four_velocity, rest_radial, rest_azimuthal = _fluid_rest_tetrad(
        geometry,
        primitive,
    )
    stress_mass = sigma * specific_stress
    stress_tensor = stress_mass * (
        np.outer(rest_radial, rest_azimuthal)
        + np.outer(rest_azimuthal, rest_radial)
    )
    metric = geometry.spacetime_metric
    alpha = geometry.base.lapse

    mixed_time = stress_tensor[0] @ metric
    mixed_radial = stress_tensor[1] @ metric
    momentum_increment = alpha * mixed_time[1:]
    killing_density_increment = -alpha * mixed_time[0]
    stress_conserved = np.asarray(
        [
            0.0,
            momentum_increment[0],
            momentum_increment[1],
            killing_density_increment,
        ],
        dtype=float,
    )
    stress_flux = np.asarray(
        [
            0.0,
            alpha * mixed_radial[1],
            alpha * mixed_radial[2],
            -alpha * mixed_radial[0],
        ],
        dtype=float,
    )
    killing_conserved = perfect.killing_conserved + stress_conserved
    killing_flux = perfect.killing_flux_over_c + stress_flux

    lower_velocity = metric @ four_velocity
    tensor_orthogonality = stress_tensor @ lower_velocity
    coordinate_radial_speed = four_velocity[1] / four_velocity[0]
    coordinate_azimuthal_speed = four_velocity[2] / four_velocity[0]
    radial_work = alpha * (
        coordinate_radial_speed * mixed_radial[1]
        + coordinate_azimuthal_speed * mixed_radial[2]
    )
    radial_source_increment = 0.5 * alpha * float(
        np.sum(
            stress_tensor
            * geometry.radial_spacetime_metric_derivative
        )
    )
    tensor_scale = max(abs(stress_mass), np.finfo(float).tiny)
    orthogonality_scale = tensor_scale * max(
        float(np.max(np.abs(lower_velocity))),
        1.0,
    )
    work_scale = max(
        abs(stress_flux[3]),
        abs(radial_work),
        tensor_scale,
    )
    rest_mass = perfect.killing_conserved[0]
    return CausalStressColumnState(
        killing_conserved=np.asarray(killing_conserved, dtype=float),
        killing_flux_over_c=np.asarray(killing_flux, dtype=float),
        stress_killing_conserved_increment=stress_conserved,
        stress_killing_flux_increment_over_c=stress_flux,
        viscous_stress_tensor=np.asarray(stress_tensor, dtype=float),
        relaxing_stress_conserved=float(rest_mass * specific_stress),
        relaxing_stress_flux_over_c=float(
            rest_mass
            * specific_stress
            * perfect.valencia_state.transport_velocity_over_c
        ),
        specific_stress=specific_stress,
        lorentz_factor=perfect.valencia_state.lorentz_factor,
        coordinate_angular_velocity=float(
            C * coordinate_azimuthal_speed
        ),
        radial_geometric_source_increment=float(radial_source_increment),
        tensor_trace_relative_defect=float(
            abs(np.sum(metric * stress_tensor)) / tensor_scale
        ),
        tensor_orthogonality_relative_defect=float(
            np.max(np.abs(tensor_orthogonality))
            / orthogonality_scale
        ),
        radial_work_relative_defect=float(
            abs(stress_flux[3] - radial_work) / work_scale
        ),
    )


def causal_stress_torque_and_power(
    geometry: KerrSchildColumnGeometry,
    state: CausalStressColumnState,
) -> tuple[float, float]:
    """Return outward torque and Killing power carried by the shear stress."""

    torque = (
        geometry.face_measure
        * C**2
        * state.stress_killing_flux_increment_over_c[2]
    )
    power = (
        geometry.face_measure
        * C**3
        * state.stress_killing_flux_increment_over_c[3]
    )
    return float(torque), float(power)


def causal_stress_relaxation_source(
    geometry: KerrSchildColumnGeometry,
    state: CausalStressColumnState,
    closure: CausalAlphaShearClosure,
    *,
    positive_shear_rate: float,
) -> float:
    """Return the local stress-density source per unit coordinate ``ct``."""

    target = closure.target_specific_stress(positive_shear_rate)
    source = (
        geometry.base.lapse
        * state.killing_conserved[0]
        / state.lorentz_factor
        * (target - state.specific_stress)
        / (C * closure.relaxation_time)
    )
    return float(source)


def _causal_shear_principal_matrix(
    closure: CausalAlphaShearClosure,
) -> np.ndarray:
    """Return the LRF transverse momentum/stress principal matrix."""

    enthalpy = closure.specific_enthalpy_over_c2
    speed_squared = closure.viscous_signal_speed_over_c**2
    return np.asarray(
        [
            [0.0, 1.0 / enthalpy],
            [enthalpy * speed_squared, 0.0],
        ],
        dtype=float,
    )


def audit_causal_stress_characteristics(
    geometry: KerrSchildColumnGeometry,
    eos: FixedHeightGasRadiationColumnEOS,
    closure: CausalAlphaShearClosure,
    *,
    surface_density: float,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
    temperature: float,
) -> CausalStressCharacteristicAudit:
    """Audit acoustic, contact, and causal transverse-shear modes."""

    thermodynamics = eos.from_surface_density_temperature(
        surface_density,
        temperature,
    )
    sound_speed = thermodynamics.sound_speed / C
    perfect_speeds = valencia_radial_characteristic_speeds_over_c(
        geometry.base,
        radial_velocity_over_c=radial_velocity_over_c,
        azimuthal_velocity_over_c=azimuthal_velocity_over_c,
        sound_speed_over_c=sound_speed,
    )
    shear_cone = valencia_radial_characteristic_speeds_over_c(
        geometry.base,
        radial_velocity_over_c=radial_velocity_over_c,
        azimuthal_velocity_over_c=azimuthal_velocity_over_c,
        sound_speed_over_c=closure.viscous_signal_speed_over_c,
    )
    acoustic = (perfect_speeds[0], perfect_speeds[-1])
    shear = (shear_cone[0], shear_cone[-1])
    advective = perfect_speeds[1]
    speeds = np.sort(np.asarray((*acoustic, *shear, advective)))

    principal = _causal_shear_principal_matrix(closure)
    principal_eigenvalues = np.linalg.eigvals(principal)
    numerical_rest_speeds = np.sort(np.real(principal_eigenvalues))
    analytic_rest_speeds = np.asarray(
        [
            -closure.viscous_signal_speed_over_c,
            closure.viscous_signal_speed_over_c,
        ]
    )
    eigenvalue_defect = float(
        np.max(np.abs(numerical_rest_speeds - analytic_rest_speeds))
    )
    maximum_imaginary = float(
        np.max(np.abs(np.imag(principal_eigenvalues)))
    )
    light_min = geometry.base.ingoing_light_speed_over_c
    light_max = geometry.base.outgoing_light_speed_over_c
    light_excess = max(
        float(light_min - np.min(speeds)),
        float(np.max(speeds) - light_max),
        0.0,
    )
    rank_threshold = max(
        float(np.max(np.abs(speeds))) * 1.0e-10,
        1.0e-12,
    )
    return CausalStressCharacteristicAudit(
        speeds_over_c=tuple(float(value) for value in speeds),
        acoustic_speeds_over_c=tuple(float(value) for value in acoustic),
        shear_speeds_over_c=tuple(float(value) for value in shear),
        local_rest_shear_speeds_over_c=tuple(
            float(value) for value in numerical_rest_speeds
        ),
        incoming_inner_characteristics=int(np.sum(speeds > 0.0)),
        stationary_flux_rank=int(
            np.sum(np.abs(speeds) > rank_threshold)
        ),
        smallest_absolute_characteristic_speed=float(
            np.min(np.abs(speeds))
        ),
        shear_principal_eigenvalue_defect=eigenvalue_defect,
        maximum_imaginary_eigenvalue=maximum_imaginary,
        maximum_light_cone_excess=float(light_excess),
    )


def audit_advected_stress_flux_eigensystem(
    geometry: KerrSchildColumnGeometry,
    eos: FixedHeightGasRadiationColumnEOS,
    closure: CausalAlphaShearClosure,
    *,
    surface_density: float,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
    temperature: float,
    finite_difference_step: float = 2.0e-4,
) -> AdvectedStressFluxAudit:
    """Audit the rejected pressure-amplitude-only stress architecture.

    This control injects the independent stress into the covariant
    stress-energy tensor and merely advects ``D chi``. A local relaxation
    toward ``alpha Pi`` changes only lower-order source terms, so this flux
    Jacobian is its complete principal part. It intentionally omits the
    shear-gradient coupling present in the accepted causal closure.
    """

    if not 0.0 < finite_difference_step < 1.0e-2:
        raise ValueError("finite-difference step must be positive and small")
    stress_scale = closure.equilibrium_specific_stress
    chart = np.asarray(
        [
            np.log(surface_density),
            radial_velocity_over_c,
            azimuthal_velocity_over_c,
            np.log(temperature),
            1.0,
        ],
        dtype=float,
    )

    def state_arrays(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local_surface_density = float(np.exp(values[0]))
        thermodynamics = eos.from_surface_density_temperature(
            local_surface_density,
            float(np.exp(values[3])),
        )
        primitive = ValenciaPerfectFluidPrimitive(
            surface_density=local_surface_density,
            radial_velocity_over_c=float(values[1]),
            azimuthal_velocity_over_c=float(values[2]),
            specific_internal_energy=(
                thermodynamics.specific_internal_energy
            ),
            integrated_pressure=thermodynamics.integrated_pressure,
        )
        state = causal_stress_column_state(
            geometry,
            primitive,
            specific_stress=float(values[4] * stress_scale),
        )
        conserved = np.concatenate(
            (
                state.killing_conserved,
                [state.relaxing_stress_conserved],
            )
        )
        flux = np.concatenate(
            (
                state.killing_flux_over_c,
                [state.relaxing_stress_flux_over_c],
            )
        )
        return conserved, flux

    conserved_jacobian = np.empty((5, 5), dtype=float)
    flux_jacobian = np.empty((5, 5), dtype=float)
    for index in range(5):
        offsets = []
        for multiplier in (-2.0, -1.0, 1.0, 2.0):
            candidate = np.array(chart, copy=True)
            candidate[index] += multiplier * finite_difference_step
            offsets.append(state_arrays(candidate))
        minus_two, minus, plus, plus_two = offsets
        conserved_jacobian[:, index] = (
            minus_two[0]
            - 8.0 * minus[0]
            + 8.0 * plus[0]
            - plus_two[0]
        ) / (12.0 * finite_difference_step)
        flux_jacobian[:, index] = (
            minus_two[1]
            - 8.0 * minus[1]
            + 8.0 * plus[1]
            - plus_two[1]
        ) / (12.0 * finite_difference_step)

    row_scale = np.maximum(
        np.max(np.abs(conserved_jacobian), axis=1),
        np.max(np.abs(flux_jacobian), axis=1),
    )
    row_scale = np.maximum(row_scale, np.max(row_scale) * 1.0e-14)
    conservative_jacobian = np.linalg.solve(
        (conserved_jacobian / row_scale[:, np.newaxis]).T,
        (flux_jacobian / row_scale[:, np.newaxis]).T,
    ).T
    eigenvalues = np.linalg.eigvals(conservative_jacobian)
    light_min = geometry.base.ingoing_light_speed_over_c
    light_max = geometry.base.outgoing_light_speed_over_c
    real_parts = np.real(eigenvalues)
    light_excess = max(
        float(light_min - np.min(real_parts)),
        float(np.max(real_parts) - light_max),
        0.0,
    )
    ordered = tuple(
        complex(value)
        for value in sorted(
            eigenvalues,
            key=lambda value: (float(np.real(value)), float(np.imag(value))),
        )
    )
    return AdvectedStressFluxAudit(
        eigenvalues=ordered,
        maximum_imaginary_eigenvalue=float(
            np.max(np.abs(np.imag(eigenvalues)))
        ),
        maximum_light_cone_excess=float(light_excess),
    )
