"""Physical entropy congruence and asymptotic-preserving port propagator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from imri_qpe.constants import C

from .causal_inner import gas_radiation_relativistic_sound_speed_squared
from .causal_inner_full_port_atlas import FullPortAtlasAnchor
from .causal_inner_geometry import KerrSchildColumnGeometry
from .causal_inner_nonlinear_port_atlas import (
    equilibrium_entropy_point_from_primitive,
    equilibrium_entropy_variables_decimal,
    equilibrium_temporal_conserved,
)
from .causal_inner_valencia import valencia_radial_characteristic_speeds_over_c


@dataclass(frozen=True)
class PhysicalEntropyCongruence:
    primitive: np.ndarray
    temporal_conserved: np.ndarray
    temporal_primitive_jacobian: np.ndarray
    radial_flux_primitive_jacobian: np.ndarray
    entropy_primitive_jacobian: np.ndarray
    conserved_scales: np.ndarray
    scaled_entropy_metric: np.ndarray
    scaled_entropy_square_root: np.ndarray
    scaled_entropy_inverse_square_root: np.ndarray
    physical_flux_jacobian: np.ndarray
    whitened_radial_matrix: np.ndarray
    numerical_speeds_over_c: np.ndarray
    analytic_speeds_over_c: np.ndarray
    sound_speed_over_c: float


@dataclass(frozen=True)
class PhysicalEntropyCongruenceAudit:
    scaled_entropy_minimum_eigenvalue_ratio: float
    whitened_symmetry_relative_defect: float
    valencia_spectrum_absolute_defect: float
    core_reconstruction_relative_defect: float
    maximum_absolute_speed_over_c: float

    @property
    def passed(self) -> bool:
        return bool(
            self.scaled_entropy_minimum_eigenvalue_ratio >= 1.0e-12
            and self.whitened_symmetry_relative_defect <= 2.0e-6
            and self.valencia_spectrum_absolute_defect <= 2.0e-6
            and self.core_reconstruction_relative_defect <= 2.0e-6
            and self.maximum_absolute_speed_over_c < 1.0
        )


@dataclass(frozen=True)
class CorrectedPhysicalPortAtlas:
    rest_matrix: np.ndarray
    radial_matrix: np.ndarray
    source_matrix: np.ndarray
    core_orientation: np.ndarray
    mapped_rest_speeds_over_c: np.ndarray
    coordinate_speeds_over_c: np.ndarray
    core_matrix: np.ndarray


@dataclass(frozen=True)
class CorrectedPhysicalPortAtlasAudit:
    radial_symmetry_defect: float
    source_entropy_positive_part: float
    lower_light_cone_violation: float
    upper_light_cone_violation: float
    core_orientation_orthogonality_defect: float
    zero_shear_core_reconstruction_defect: float

    @property
    def passed(self) -> bool:
        return bool(
            self.radial_symmetry_defect <= 2.0e-12
            and self.source_entropy_positive_part <= 2.0e-12
            and self.lower_light_cone_violation <= 2.0e-12
            and self.upper_light_cone_violation <= 2.0e-12
            and self.core_orientation_orthogonality_defect <= 2.0e-12
            and self.zero_shear_core_reconstruction_defect <= 2.0e-6
        )


@dataclass(frozen=True)
class APFastPropagatorAudit:
    maximum_semigroup_expansivity: float
    maximum_composition_defect: float
    stiff_limit_defect: float
    stable_spectral_gap: float

    @property
    def passed(self) -> bool:
        return bool(
            self.maximum_semigroup_expansivity <= 1.0e-10
            and self.maximum_composition_defect <= 2.0e-11
            and self.stiff_limit_defect <= 2.0e-8
            and self.stable_spectral_gap > 0.0
        )


def _primitive_evaluation(
    geometry: KerrSchildColumnGeometry,
    height: float,
    primitive: np.ndarray,
):
    point = equilibrium_entropy_point_from_primitive(
        geometry,
        density=float(np.exp(primitive[0])),
        temperature=float(np.exp(primitive[1])),
        proper_half_thickness=height,
        radial_velocity_over_c=float(primitive[2]),
        azimuthal_velocity_over_c=float(primitive[3]),
    )
    temporal = equilibrium_temporal_conserved(point)
    radial = np.asarray(
        (
            point.state.surface_mass_current[1],
            *point.state.column_stress_energy[1, (0, 1, 2)],
        ),
        dtype=float,
    )
    entropy = np.asarray(
        [float(value) for value in equilibrium_entropy_variables_decimal(point)],
        dtype=float,
    )
    return point, temporal, radial, entropy


def _fourth_order_jacobians(evaluate, primitive, step):
    collections = ([], [], [])
    for component in range(4):
        direction = np.zeros(4)
        direction[component] = step
        values = [evaluate(primitive + multiple * direction)[1:] for multiple in (-2, -1, 1, 2)]
        for quantity, columns in enumerate(collections):
            derivative = (
                -values[3][quantity]
                + 8.0 * values[2][quantity]
                - 8.0 * values[1][quantity]
                + values[0][quantity]
            ) / (12.0 * step)
            columns.append(derivative)
    return tuple(np.asarray(columns).T for columns in collections)


def build_physical_entropy_congruence(
    geometry: KerrSchildColumnGeometry,
    *,
    proper_half_thickness: float,
    density: float,
    temperature: float,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
    primitive_step: float = 3.0e-4,
) -> PhysicalEntropyCongruence:
    primitive = np.asarray(
        (
            np.log(float(density)),
            np.log(float(temperature)),
            float(radial_velocity_over_c),
            float(azimuthal_velocity_over_c),
        )
    )
    height = float(proper_half_thickness)
    evaluate = lambda value: _primitive_evaluation(geometry, height, value)
    point, temporal, _, _ = evaluate(primitive)
    temporal_q, radial_q, entropy_q = _fourth_order_jacobians(
        evaluate, primitive, float(primitive_step)
    )
    temporal_inverse = np.linalg.inv(temporal_q)
    flux_jacobian = radial_q @ temporal_inverse
    entropy_metric = entropy_q @ temporal_inverse
    scales = np.maximum(
        np.maximum(np.abs(temporal), np.max(np.abs(temporal_q), axis=1)), 1.0
    )
    scaling = np.diag(scales)
    scaled_metric = scaling @ entropy_metric @ scaling
    scaled_metric = 0.5 * (scaled_metric + scaled_metric.T)
    eigenvalues = np.linalg.eigvalsh(scaled_metric)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("physical entropy metric is not positive definite")
    # If G=L L^T, w=L^T y makes the entropy norm Euclidean.  The Cholesky
    # congruence is materially more accurate than an eigensquare-root for the
    # strongly conditioned rest-mass/thermal metric.
    square_root = np.linalg.cholesky(scaled_metric).T
    inverse_square_root = np.linalg.inv(square_root)
    scaled_flux = np.linalg.solve(scaling, flux_jacobian @ scaling)
    whitened = square_root @ scaled_flux @ inverse_square_root
    symmetric_whitened = 0.5 * (whitened + whitened.T)
    numerical = np.linalg.eigvalsh(symmetric_whitened)
    sound = float(
        np.sqrt(
            gas_radiation_relativistic_sound_speed_squared(
                point.state.density, point.state.temperature
            )
        )
        / C
    )
    analytic = np.asarray(
        valencia_radial_characteristic_speeds_over_c(
            geometry.base,
            radial_velocity_over_c=radial_velocity_over_c,
            azimuthal_velocity_over_c=azimuthal_velocity_over_c,
            sound_speed_over_c=sound,
        )
    )
    return PhysicalEntropyCongruence(
        primitive,
        temporal,
        temporal_q,
        radial_q,
        entropy_q,
        scales,
        scaled_metric,
        square_root,
        inverse_square_root,
        flux_jacobian,
        whitened,
        numerical,
        analytic,
        sound,
    )


def audit_physical_entropy_congruence(
    congruence: PhysicalEntropyCongruence,
) -> PhysicalEntropyCongruenceAudit:
    eigenvalues = np.linalg.eigvalsh(congruence.scaled_entropy_metric)
    raw_scaled_flux = np.linalg.solve(
        np.diag(congruence.conserved_scales),
        congruence.physical_flux_jacobian @ np.diag(congruence.conserved_scales),
    )
    raw_whitened = (
        congruence.scaled_entropy_square_root
        @ raw_scaled_flux
        @ congruence.scaled_entropy_inverse_square_root
    )
    symmetry = float(
        np.linalg.norm(raw_whitened - raw_whitened.T)
        / max(np.linalg.norm(raw_whitened), np.finfo(float).tiny)
    )
    reconstructed_scaled = (
        congruence.scaled_entropy_inverse_square_root
        @ congruence.whitened_radial_matrix
        @ congruence.scaled_entropy_square_root
    )
    reconstruction = float(
        np.linalg.norm(reconstructed_scaled - raw_scaled_flux)
        / max(np.linalg.norm(raw_scaled_flux), np.finfo(float).tiny)
    )
    return PhysicalEntropyCongruenceAudit(
        float(np.min(eigenvalues) / np.max(eigenvalues)),
        symmetry,
        float(
            np.max(
                np.abs(
                    congruence.numerical_speeds_over_c
                    - congruence.analytic_speeds_over_c
                )
            )
        ),
        reconstruction,
        float(np.max(np.abs(congruence.numerical_speeds_over_c))),
    )


def kerr_schild_signal_speed_map(
    geometry: KerrSchildColumnGeometry,
    *,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
    rest_speed_over_c: float,
) -> float:
    radial = float(radial_velocity_over_c)
    azimuthal = float(azimuthal_velocity_over_c)
    signal = float(rest_speed_over_c)
    speed_squared = radial**2 + azimuthal**2
    signal_squared = signal**2
    coordinate_radial = radial / np.sqrt(geometry.base.gamma_rr)
    denominator = 1.0 - speed_squared * signal_squared
    radicand = (1.0 - speed_squared) * (
        geometry.base.inverse_gamma_rr * (1.0 - speed_squared * signal_squared)
        - coordinate_radial**2 * (1.0 - signal_squared)
    )
    if denominator <= 0.0 or radicand < -1.0e-14:
        raise ValueError("signal map left the causal state domain")
    return float(
        geometry.base.lapse
        * (
            coordinate_radial * (1.0 - signal_squared)
            + signal * np.sqrt(max(radicand, 0.0))
        )
        / denominator
        - geometry.base.radial_shift_over_c
    )


def _spectral_signal_map(matrix, geometry, radial, azimuthal):
    speeds, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    mapped = np.asarray(
        [
            kerr_schild_signal_speed_map(
                geometry,
                radial_velocity_over_c=radial,
                azimuthal_velocity_over_c=azimuthal,
                rest_speed_over_c=speed,
            )
            for speed in speeds
        ]
    )
    return (vectors * mapped) @ vectors.T, speeds, mapped


def build_corrected_physical_port_atlas(
    anchor: FullPortAtlasAnchor,
    congruence: PhysicalEntropyCongruence,
    geometry: KerrSchildColumnGeometry,
) -> CorrectedPhysicalPortAtlas:
    radial = float(congruence.primitive[2])
    azimuthal = float(congruence.primitive[3])
    rest_matrix = np.array(anchor.rest_radial_matrix, copy=True)
    sound_ratio = congruence.sound_speed_over_c / anchor.sound_speed_over_c
    rest_matrix[0, 1] = rest_matrix[1, 0] = congruence.sound_speed_over_c
    rest_matrix[1:3, 4:9] *= sound_ratio
    rest_matrix[4:9, 1:3] *= sound_ratio
    rest_matrix[10, 4:9] *= sound_ratio
    rest_matrix[4:9, 10] *= sound_ratio
    mapped_full, rest_speeds, coordinate_speeds = _spectral_signal_map(
        rest_matrix, geometry, radial, azimuthal
    )
    mapped_core, _, _ = _spectral_signal_map(
        rest_matrix[:4, :4], geometry, radial, azimuthal
    )
    _, abstract_vectors = np.linalg.eigh(mapped_core)
    physical_core = 0.5 * (
        congruence.whitened_radial_matrix
        + congruence.whitened_radial_matrix.T
    )
    _, physical_vectors = np.linalg.eigh(physical_core)
    orientation = physical_vectors @ abstract_vectors.T
    embedding = np.eye(11)
    embedding[:4, :4] = orientation
    radial_matrix = embedding @ mapped_full @ embedding.T
    source_matrix = embedding @ anchor.source_matrix @ embedding.T
    return CorrectedPhysicalPortAtlas(
        rest_matrix,
        0.5 * (radial_matrix + radial_matrix.T),
        source_matrix,
        orientation,
        rest_speeds,
        coordinate_speeds,
        physical_core,
    )


def audit_corrected_physical_port_atlas(
    atlas: CorrectedPhysicalPortAtlas,
    anchor: FullPortAtlasAnchor,
    congruence: PhysicalEntropyCongruence,
    geometry: KerrSchildColumnGeometry,
) -> CorrectedPhysicalPortAtlasAudit:
    radial = float(congruence.primitive[2])
    azimuthal = float(congruence.primitive[3])
    mapped_core, _, _ = _spectral_signal_map(
        atlas.rest_matrix[:4, :4], geometry, radial, azimuthal
    )
    reconstructed_core = atlas.core_orientation @ mapped_core @ atlas.core_orientation.T
    speeds = np.linalg.eigvalsh(atlas.radial_matrix)
    source_symmetric = 0.5 * (atlas.source_matrix + atlas.source_matrix.T)
    return CorrectedPhysicalPortAtlasAudit(
        float(np.linalg.norm(atlas.radial_matrix - atlas.radial_matrix.T)),
        max(float(np.max(np.linalg.eigvalsh(source_symmetric))), 0.0),
        max(float(geometry.base.ingoing_light_speed_over_c - np.min(speeds)), 0.0),
        max(float(np.max(speeds) - geometry.base.outgoing_light_speed_over_c), 0.0),
        float(np.linalg.norm(atlas.core_orientation.T @ atlas.core_orientation - np.eye(4))),
        float(
            np.linalg.norm(reconstructed_core - congruence.whitened_radial_matrix)
            / max(np.linalg.norm(congruence.whitened_radial_matrix), np.finfo(float).tiny)
        ),
    )


def exponential_affine_step(generator, state, forcing, timestep):
    matrix = np.asarray(generator)
    value = np.asarray(state)
    drive = np.asarray(forcing)
    count = matrix.shape[0]
    augmented = np.zeros((count + 1, count + 1), dtype=np.result_type(matrix, value, drive))
    augmented[:count, :count] = float(timestep) * matrix
    augmented[:count, count] = float(timestep) * drive
    result = expm(augmented) @ np.concatenate((value, np.ones(1, dtype=value.dtype)))
    return result[:count]


def audit_ap_fast_propagator(
    atlas: CorrectedPhysicalPortAtlas,
    *,
    step_ratios=(1.0e-3, 1.0, 1.0e3),
) -> APFastPropagatorAudit:
    source_eigenvalues = np.linalg.eigvals(atlas.source_matrix)
    stable = -np.real(source_eigenvalues[np.real(source_eigenvalues) < -1.0e-12])
    if stable.size == 0:
        return APFastPropagatorAudit(float("inf"), float("inf"), float("inf"), 0.0)
    gap = float(np.min(stable))
    wave_frequency = gap
    generator = -1j * wave_frequency * atlas.radial_matrix + atlas.source_matrix
    expansivity = 0.0
    composition = 0.0
    for ratio in step_ratios:
        timestep = float(ratio) / gap
        full = expm(timestep * generator)
        half = expm(0.5 * timestep * generator)
        expansivity = max(expansivity, max(float(np.linalg.svd(full, compute_uv=False)[0] - 1.0), 0.0))
        composition = max(
            composition,
            float(np.linalg.norm(full - half @ half) / max(np.linalg.norm(full), 1.0)),
        )
    projector = np.zeros((11, 11))
    projector[:4, :4] = np.eye(4)
    stiff = expm((1.0e3 / gap) * atlas.source_matrix)
    stiff_defect = float(np.linalg.norm(stiff - projector))
    return APFastPropagatorAudit(expansivity, composition, stiff_defect, gap)


__all__ = [
    "APFastPropagatorAudit",
    "CorrectedPhysicalPortAtlas",
    "CorrectedPhysicalPortAtlasAudit",
    "PhysicalEntropyCongruence",
    "PhysicalEntropyCongruenceAudit",
    "audit_ap_fast_propagator",
    "audit_corrected_physical_port_atlas",
    "audit_physical_entropy_congruence",
    "build_corrected_physical_port_atlas",
    "build_physical_entropy_congruence",
    "exponential_affine_step",
    "kerr_schild_signal_speed_map",
]
