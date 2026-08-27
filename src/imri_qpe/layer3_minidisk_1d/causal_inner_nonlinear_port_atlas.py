"""Nonlinear equilibrium entropy paths and moving-STF atlas connections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from imri_qpe.constants import C, DEFAULT_MU_MOL

from .causal_inner_eleven_field_convex import FullShearRestFrame, full_shear_rest_frame
from .causal_inner_equilibrium_potential import (
    CompensatedMassAffinity,
    EquilibriumColumnPotentialState,
    entropy_variables_from_primitive,
    equilibrium_column_potential_state,
)
from .causal_inner_geometry import KerrSchildColumnGeometry


_ENTROPY_BETA_INDICES = (0, 1, 2)
_RADIAL_INDEX = 1


@dataclass(frozen=True)
class EquilibriumEntropyPoint:
    metric: np.ndarray
    inverse_metric: np.ndarray
    mass_affinity: CompensatedMassAffinity
    inverse_temperature_covector: np.ndarray
    proper_half_thickness: float
    state: EquilibriumColumnPotentialState


@dataclass(frozen=True)
class EntropyPathFluxAudit:
    tadmor_relative_defect: float
    quadrature_refinement_relative_defect: float
    minimum_path_temperature: float
    minimum_path_density: float
    node_count: int

    @property
    def passed(self) -> bool:
        return (
            self.tadmor_relative_defect <= 2.0e-9
            and self.quadrature_refinement_relative_defect <= 2.0e-9
            and self.minimum_path_temperature > 0.0
            and self.minimum_path_density > 0.0
            and self.node_count == 16
        )


@dataclass(frozen=True)
class STFPolarConnection:
    left_to_right: np.ndarray
    singular_values: np.ndarray
    polar_stretch: float


@dataclass(frozen=True)
class STFPolarConnectionAudit:
    orthogonality_defect: float
    reverse_roundtrip_defect: float
    polar_stretch: float
    determinant: float

    @property
    def passed(self) -> bool:
        return (
            self.orthogonality_defect <= 2.0e-12
            and self.reverse_roundtrip_defect <= 2.0e-12
            and self.polar_stretch <= 0.08
            and self.determinant > 0.0
        )


@dataclass(frozen=True)
class ConditionedDiscreteGradientFlux:
    flux: np.ndarray
    base_flux: np.ndarray
    correction: np.ndarray
    flux_scales: np.ndarray


@dataclass(frozen=True)
class ConditionedDiscreteGradientFluxAudit:
    tadmor_relative_defect: float
    swap_symmetry_relative_defect: float
    endpoint_consistency_relative_defect: float
    weighted_correction_relative_norm: float
    entropy_penalty_positive_part: float

    @property
    def passed(self) -> bool:
        return (
            self.tadmor_relative_defect <= 2.0e-12
            and self.swap_symmetry_relative_defect <= 2.0e-13
            and self.endpoint_consistency_relative_defect <= 2.0e-13
            and self.weighted_correction_relative_norm <= 0.05
            and self.entropy_penalty_positive_part <= 0.0
        )


def equilibrium_entropy_point_from_primitive(
    geometry: KerrSchildColumnGeometry,
    *,
    density: float,
    temperature: float,
    proper_half_thickness: float,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
    vertical_velocity_over_c: float = 0.0,
    mu_mol: float = DEFAULT_MU_MOL,
) -> EquilibriumEntropyPoint:
    frame = full_shear_rest_frame(
        geometry,
        radial_velocity_over_c=radial_velocity_over_c,
        azimuthal_velocity_over_c=azimuthal_velocity_over_c,
        vertical_velocity_over_c=vertical_velocity_over_c,
    )
    affinity, beta = entropy_variables_from_primitive(
        frame.metric,
        frame.four_velocity,
        density=density,
        temperature=temperature,
        mu_mol=mu_mol,
    )
    state = equilibrium_column_potential_state(
        frame.metric,
        affinity,
        beta,
        proper_half_thickness=proper_half_thickness,
        mu_mol=mu_mol,
    )
    return EquilibriumEntropyPoint(
        np.asarray(frame.metric, dtype=float),
        np.asarray(frame.inverse_metric, dtype=float),
        affinity,
        np.asarray(beta, dtype=float),
        float(proper_half_thickness),
        state,
    )


def _require_compatible_points(
    left: EquilibriumEntropyPoint, right: EquilibriumEntropyPoint
) -> None:
    if not isinstance(left, EquilibriumEntropyPoint) or not isinstance(
        right, EquilibriumEntropyPoint
    ):
        raise TypeError("entropy path endpoints must be EquilibriumEntropyPoint")
    if not np.array_equal(left.metric, right.metric):
        raise ValueError("entropy path requires one frozen metric")
    if left.proper_half_thickness != right.proper_half_thickness:
        raise ValueError("equilibrium entropy path requires one frozen height")
    if left.inverse_temperature_covector[3] != right.inverse_temperature_covector[3]:
        raise ValueError("beta_z belongs to the separate height port")


def _path_point(
    left: EquilibriumEntropyPoint,
    right: EquilibriumEntropyPoint,
    fraction: float,
) -> EquilibriumEntropyPoint:
    _require_compatible_points(left, right)
    s = np.longdouble(fraction)
    one_minus = np.longdouble(1.0) - s
    beta = one_minus * np.asarray(
        left.inverse_temperature_covector, dtype=np.longdouble
    ) + s * np.asarray(right.inverse_temperature_covector, dtype=np.longdouble)
    inverse_metric = np.asarray(left.inverse_metric, dtype=np.longdouble)
    temperature = np.longdouble(1.0) / np.sqrt(-(beta @ inverse_metric @ beta))
    left_temperature = np.longdouble(left.state.temperature)
    right_temperature = np.longdouble(right.state.temperature)
    left_thermal = np.longdouble(left.mass_affinity.thermal_part)
    right_thermal = np.longdouble(right.mass_affinity.thermal_part)
    c_squared = np.longdouble(C) ** 2
    thermal = c_squared * (
        one_minus / left_temperature
        + s / right_temperature
        - np.longdouble(1.0) / temperature
    ) + one_minus * left_thermal + s * right_thermal
    affinity = CompensatedMassAffinity(
        rest_mass_part=float(c_squared / temperature),
        thermal_part=float(thermal),
    )
    state = equilibrium_column_potential_state(
        left.metric,
        affinity,
        np.asarray(beta, dtype=float),
        proper_half_thickness=left.proper_half_thickness,
    )
    return EquilibriumEntropyPoint(
        left.metric,
        left.inverse_metric,
        affinity,
        np.asarray(beta, dtype=float),
        left.proper_half_thickness,
        state,
    )


def _radial_current_gradient(point: EquilibriumEntropyPoint) -> np.ndarray:
    state = point.state
    return np.asarray(
        (
            state.surface_mass_current[_RADIAL_INDEX],
            *state.column_stress_energy[_RADIAL_INDEX, _ENTROPY_BETA_INDICES],
        ),
        dtype=float,
    )


def equilibrium_entropy_conservative_radial_flux(
    left: EquilibriumEntropyPoint,
    right: EquilibriumEntropyPoint,
    *,
    node_count: int = 16,
) -> tuple[np.ndarray, tuple[EquilibriumEntropyPoint, ...]]:
    _require_compatible_points(left, right)
    nodes, weights = leggauss(int(node_count))
    fractions = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    points = tuple(_path_point(left, right, float(value)) for value in fractions)
    flux = sum(
        float(weight) * _radial_current_gradient(point)
        for weight, point in zip(weights, points, strict=True)
    )
    return np.asarray(flux, dtype=float), points


def _entropy_jump(left: EquilibriumEntropyPoint, right: EquilibriumEntropyPoint) -> np.ndarray:
    c_squared = np.longdouble(C) ** 2
    alpha_jump = c_squared * (
        np.longdouble(1.0) / np.longdouble(right.state.temperature)
        - np.longdouble(1.0) / np.longdouble(left.state.temperature)
    ) + np.longdouble(right.mass_affinity.thermal_part) - np.longdouble(
        left.mass_affinity.thermal_part
    )
    beta_jump = np.asarray(
        right.inverse_temperature_covector - left.inverse_temperature_covector,
        dtype=np.longdouble,
    )
    return np.asarray((alpha_jump, *beta_jump[list(_ENTROPY_BETA_INDICES)]), dtype=np.longdouble)


def conditioned_discrete_gradient_radial_flux(
    left: EquilibriumEntropyPoint,
    right: EquilibriumEntropyPoint,
) -> ConditionedDiscreteGradientFlux:
    """Return a symmetric weighted discrete gradient of ``X^R``.

    No interior entropy-path state is evaluated.  The rank-one correction is
    the minimum-norm correction in endpoint-flux-scaled coordinates subject
    to the exact discrete chain rule.
    """

    _require_compatible_points(left, right)
    left_flux = _radial_current_gradient(left)
    right_flux = _radial_current_gradient(right)
    base = 0.5 * (left_flux + right_flux)
    largest = max(float(np.max(np.abs(left_flux))), float(np.max(np.abs(right_flux))), 1.0)
    scales = np.maximum(np.maximum(np.abs(left_flux), np.abs(right_flux)), largest * 1.0e-14)
    jump = _entropy_jump(left, right)
    potential_jump = np.longdouble(right.state.potential_current[_RADIAL_INDEX]) - np.longdouble(left.state.potential_current[_RADIAL_INDEX])
    residual = potential_jump - np.sum(jump * np.asarray(base, dtype=np.longdouble))
    dual = np.asarray(scales, dtype=np.longdouble) ** 2 * jump
    denominator = np.sum(jump * dual)
    if denominator == 0.0:
        correction = np.zeros_like(base)
    else:
        correction = np.asarray(residual * dual / denominator, dtype=float)
    return ConditionedDiscreteGradientFlux(base + correction, base, correction, scales)


def audit_conditioned_discrete_gradient_radial_flux(
    left: EquilibriumEntropyPoint,
    right: EquilibriumEntropyPoint,
) -> ConditionedDiscreteGradientFluxAudit:
    forward = conditioned_discrete_gradient_radial_flux(left, right)
    reverse = conditioned_discrete_gradient_radial_flux(right, left)
    consistent = conditioned_discrete_gradient_radial_flux(left, left)
    physical_left = _radial_current_gradient(left)
    jump = _entropy_jump(left, right)
    potential_jump = np.longdouble(right.state.potential_current[_RADIAL_INDEX]) - np.longdouble(left.state.potential_current[_RADIAL_INDEX])
    contraction = np.sum(jump * np.asarray(forward.flux, dtype=np.longdouble))
    tadmor_scale = max(float(abs(contraction)), float(abs(potential_jump)), np.finfo(float).tiny)
    flux_scale = max(float(np.linalg.norm(forward.flux)), float(np.linalg.norm(reverse.flux)), np.finfo(float).tiny)
    consistency_scale = max(float(np.linalg.norm(physical_left)), np.finfo(float).tiny)
    weighted_correction = float(
        np.linalg.norm(forward.correction / forward.flux_scales)
        / max(np.linalg.norm(forward.base_flux / forward.flux_scales), 1.0)
    )
    penalty_contraction = -0.5 * np.sum(
        jump * (np.asarray(forward.flux_scales, dtype=np.longdouble) ** 2 * jump)
    )
    return ConditionedDiscreteGradientFluxAudit(
        float(abs(contraction - potential_jump) / tadmor_scale),
        float(np.linalg.norm(forward.flux - reverse.flux) / flux_scale),
        float(np.linalg.norm(consistent.flux - physical_left) / consistency_scale),
        weighted_correction,
        max(float(penalty_contraction), 0.0),
    )


def audit_equilibrium_entropy_path_flux(
    left: EquilibriumEntropyPoint, right: EquilibriumEntropyPoint
) -> EntropyPathFluxAudit:
    flux16, points = equilibrium_entropy_conservative_radial_flux(
        left, right, node_count=16
    )
    flux8, _ = equilibrium_entropy_conservative_radial_flux(
        left, right, node_count=8
    )
    jump = _entropy_jump(left, right)
    lhs = np.sum(jump * np.asarray(flux16, dtype=np.longdouble))
    rhs = np.longdouble(right.state.potential_current[_RADIAL_INDEX]) - np.longdouble(
        left.state.potential_current[_RADIAL_INDEX]
    )
    scale = max(float(abs(lhs)), float(abs(rhs)), np.finfo(float).tiny)
    tadmor = float(abs(lhs - rhs) / scale)
    refinement = float(
        np.linalg.norm(flux16 - flux8)
        / max(np.linalg.norm(flux16), np.linalg.norm(flux8), np.finfo(float).tiny)
    )
    return EntropyPathFluxAudit(
        tadmor,
        refinement,
        float(min(point.state.temperature for point in points)),
        float(min(point.state.density for point in points)),
        16,
    )


def _cross_gram(left: FullShearRestFrame, right: FullShearRestFrame) -> np.ndarray:
    if not np.array_equal(left.metric, right.metric):
        raise ValueError("STF connection requires one frozen metric")
    lowered_left = np.einsum(
        "ik,akl,lj->aij", left.metric, left.stf_basis, left.metric
    )
    return np.einsum("aij,bij->ab", lowered_left, right.stf_basis)


def stf_polar_connection(
    left: FullShearRestFrame, right: FullShearRestFrame
) -> STFPolarConnection:
    cross = _cross_gram(left, right).T
    left_singular, singular_values, right_singular = np.linalg.svd(cross)
    connection = left_singular @ right_singular
    return STFPolarConnection(
        connection,
        singular_values,
        float(np.max(np.abs(singular_values - 1.0))),
    )


def audit_stf_polar_connection(
    left: FullShearRestFrame, right: FullShearRestFrame
) -> STFPolarConnectionAudit:
    forward = stf_polar_connection(left, right)
    reverse = stf_polar_connection(right, left)
    identity = np.eye(5)
    return STFPolarConnectionAudit(
        float(np.linalg.norm(forward.left_to_right.T @ forward.left_to_right - identity)),
        float(np.linalg.norm(reverse.left_to_right @ forward.left_to_right - identity)),
        forward.polar_stretch,
        float(np.linalg.det(forward.left_to_right)),
    )


__all__ = [
    "ConditionedDiscreteGradientFlux",
    "ConditionedDiscreteGradientFluxAudit",
    "EntropyPathFluxAudit",
    "EquilibriumEntropyPoint",
    "STFPolarConnection",
    "STFPolarConnectionAudit",
    "audit_equilibrium_entropy_path_flux",
    "audit_conditioned_discrete_gradient_radial_flux",
    "audit_stf_polar_connection",
    "equilibrium_entropy_conservative_radial_flux",
    "equilibrium_entropy_point_from_primitive",
    "conditioned_discrete_gradient_radial_flux",
    "stf_polar_connection",
]
