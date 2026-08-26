"""Structural kernel for the eleven-field full-shear column architecture.

This module proves two prerequisites without claiming a physical nonlinear
closure:

* the five-dimensional spatial symmetric-tracefree (STF) shear space is
  represented in the instantaneous four-dimensional fluid rest frame; and
* a local eleven-field quadratic common-potential normal form is symmetric
  hyperbolic, causal for its declared fixture, and entropy dissipative.

The physical master potential, its Kerr--Schild column reduction, and all
transport coefficients remain separate prospective derivations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_inner_geometry import KerrSchildColumnGeometry


FULL_SHEAR_AMPLITUDE_NAMES = (
    "zeta_RR_minus_phiphi",
    "zeta_RR_plus_phiphi_minus_2zz",
    "zeta_Rphi",
    "zeta_Rz",
    "zeta_phiz",
)

ELEVEN_FIELD_PERTURBATION_NAMES = (
    "log_surface_density",
    "radial_velocity_over_c",
    "azimuthal_velocity_over_c",
    "log_temperature",
    "log_height",
    "vertical_velocity_over_c",
    *FULL_SHEAR_AMPLITUDE_NAMES,
)

VERTICAL_EQUILIBRIUM_NINE_FIELD_NAMES = (
    *ELEVEN_FIELD_PERTURBATION_NAMES[:4],
    *FULL_SHEAR_AMPLITUDE_NAMES,
)

N_FULL_SHEAR_AMPLITUDES = len(FULL_SHEAR_AMPLITUDE_NAMES)
N_ELEVEN_FIELDS = len(ELEVEN_FIELD_PERTURBATION_NAMES)
N_VERTICAL_EQUILIBRIUM_FIELDS = len(VERTICAL_EQUILIBRIUM_NINE_FIELD_NAMES)


@dataclass(frozen=True)
class FullShearRestFrame:
    """Four-velocity, rest triad, metric, and orthonormal STF basis."""

    metric: np.ndarray
    inverse_metric: np.ndarray
    four_velocity: np.ndarray
    rest_triad: np.ndarray
    stf_basis: np.ndarray


@dataclass(frozen=True)
class FullShearRestFrameAudit:
    """Constraint and roundtrip defects for a full-shear rest frame."""

    four_velocity_normalization_defect: float
    triad_orthonormality_defect: float
    triad_velocity_orthogonality_defect: float
    basis_symmetry_defect: float
    basis_trace_defect: float
    basis_velocity_orthogonality_defect: float
    basis_gram_defect: float
    amplitude_roundtrip_defect: float
    one_Rphi_embedding_defect: float

    @property
    def passed(self) -> bool:
        return max(self.__dict__.values()) <= 2.0e-13


@dataclass(frozen=True)
class ElevenFieldConvexNormalFormParameters:
    """Positive metric, reciprocal flux, and relaxation coefficients."""

    transport_speed_over_c: float
    surface_density_weight: float
    radial_velocity_weight: float
    azimuthal_velocity_weight: float
    thermal_weight: float
    vertical_velocity_weight: float
    shear_weights: tuple[float, float, float, float, float]
    vertical_frequency: float
    vertical_damping: float
    shear_relaxation_rates: tuple[float, float, float, float, float]
    height_log_surface_density: float
    height_log_temperature: float
    mass_radial_coupling: float
    radial_thermal_coupling: float
    radial_height_coupling: float
    thermal_height_coupling: float
    radial_shear_diagonal_couplings: tuple[float, float]
    azimuthal_Rphi_coupling: float
    vertical_Rz_coupling: float

    def __post_init__(self) -> None:
        scalar_values = (
            self.transport_speed_over_c,
            self.surface_density_weight,
            self.radial_velocity_weight,
            self.azimuthal_velocity_weight,
            self.thermal_weight,
            self.vertical_velocity_weight,
            self.vertical_frequency,
            self.vertical_damping,
            self.height_log_surface_density,
            self.height_log_temperature,
            self.mass_radial_coupling,
            self.radial_thermal_coupling,
            self.radial_height_coupling,
            self.thermal_height_coupling,
            *self.shear_weights,
            *self.shear_relaxation_rates,
            *self.radial_shear_diagonal_couplings,
            self.azimuthal_Rphi_coupling,
            self.vertical_Rz_coupling,
        )
        if any(not np.isfinite(value) for value in scalar_values):
            raise ValueError("eleven-field coefficients must be finite")
        positive = (
            self.surface_density_weight,
            self.radial_velocity_weight,
            self.azimuthal_velocity_weight,
            self.thermal_weight,
            self.vertical_velocity_weight,
            self.vertical_frequency,
            *self.shear_weights,
            *self.shear_relaxation_rates,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("entropy weights and relaxation scales must be positive")
        if self.vertical_damping < 0.0:
            raise ValueError("vertical damping must be non-negative")
        if len(self.shear_weights) != N_FULL_SHEAR_AMPLITUDES:
            raise ValueError("exactly five shear weights are required")
        if len(self.shear_relaxation_rates) != N_FULL_SHEAR_AMPLITUDES:
            raise ValueError("exactly five shear relaxation rates are required")
        if len(self.radial_shear_diagonal_couplings) != 2:
            raise ValueError("two diagonal radial-shear couplings are required")


@dataclass(frozen=True)
class ElevenFieldConvexNormalForm:
    """Hessians and source of the quadratic common-potential fixture."""

    chart_to_relaxation: np.ndarray
    entropy_metric_relaxation: np.ndarray
    flux_hessian_relaxation: np.ndarray
    source_generator_relaxation: np.ndarray
    temporal_matrix: np.ndarray
    radial_matrix: np.ndarray
    source_matrix: np.ndarray
    vertical_equilibrium_embedding: np.ndarray
    reduced_temporal_matrix: np.ndarray
    reduced_radial_matrix: np.ndarray

    def state_potential(self, chart) -> float:
        values = _require_vector(chart, N_ELEVEN_FIELDS, "chart")
        return float(0.5 * values @ self.temporal_matrix @ values)

    def radial_flux_potential(self, chart) -> float:
        values = _require_vector(chart, N_ELEVEN_FIELDS, "chart")
        return float(0.5 * values @ self.radial_matrix @ values)

    def state_current(self, chart) -> np.ndarray:
        values = _require_vector(chart, N_ELEVEN_FIELDS, "chart")
        return self.temporal_matrix @ values

    def radial_current(self, chart) -> np.ndarray:
        values = _require_vector(chart, N_ELEVEN_FIELDS, "chart")
        return self.radial_matrix @ values


@dataclass(frozen=True)
class ElevenFieldConvexNormalFormAudit:
    """Algebraic proof obligations for the quadratic fixture."""

    temporal_minimum_eigenvalue: float
    temporal_condition_number: float
    temporal_symmetry_defect: float
    radial_symmetry_defect: float
    state_gradient_defect: float
    radial_gradient_defect: float
    generalized_eigenpair_defect: float
    energy_metric_orthogonality_defect: float
    maximum_absolute_characteristic_speed_over_c: float
    source_entropy_positive_part: float
    vertical_hamiltonian_entropy_defect: float
    vertical_equilibrium_embedding_defect: float
    reduced_temporal_minimum_eigenvalue: float
    reduced_generalized_eigenpair_defect: float
    subcharacteristic_interlacing_violation: float
    full_characteristic_speeds_over_c: tuple[float, ...]
    reduced_characteristic_speeds_over_c: tuple[float, ...]

    @property
    def passed(self) -> bool:
        return (
            self.temporal_minimum_eigenvalue > 0.0
            and self.reduced_temporal_minimum_eigenvalue > 0.0
            and self.temporal_symmetry_defect <= 1.0e-13
            and self.radial_symmetry_defect <= 1.0e-13
            and self.state_gradient_defect <= 1.0e-13
            and self.radial_gradient_defect <= 1.0e-13
            and self.generalized_eigenpair_defect <= 1.0e-11
            and self.energy_metric_orthogonality_defect <= 1.0e-11
            and self.maximum_absolute_characteristic_speed_over_c < 1.0
            and self.source_entropy_positive_part <= 1.0e-12
            and self.vertical_hamiltonian_entropy_defect <= 1.0e-12
            and self.vertical_equilibrium_embedding_defect <= 1.0e-12
            and self.reduced_generalized_eigenpair_defect <= 1.0e-11
            and self.subcharacteristic_interlacing_violation <= 1.0e-12
        )


def _require_vector(values, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite and length {size}")
    return array


def _outer_symmetric(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.outer(first, second) + np.outer(second, first)


def full_shear_rest_frame(
    geometry: KerrSchildColumnGeometry,
    *,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
    vertical_velocity_over_c: float,
) -> FullShearRestFrame:
    """Construct the moving 3+1 rest triad and its five STF tensors.

    The vertical speed is measured relative to the horizontally comoving
    frame.  This exact rapidity composition is part of the new architecture;
    its small-vertical-speed limit agrees with the prior column variable.
    """

    beta_r = float(radial_velocity_over_c)
    beta_phi = float(azimuthal_velocity_over_c)
    beta_z = float(vertical_velocity_over_c)
    if any(not np.isfinite(value) for value in (beta_r, beta_phi, beta_z)):
        raise ValueError("rest-frame velocities must be finite")
    if beta_r**2 + beta_phi**2 >= 1.0 or abs(beta_z) >= 1.0:
        raise ValueError("rest-frame velocities must be subluminal")

    metric = np.zeros((4, 4), dtype=float)
    metric[:3, :3] = geometry.spacetime_metric
    metric[3, 3] = 1.0
    inverse_metric = np.linalg.inv(metric)
    normal = np.asarray(
        (
            1.0 / geometry.base.lapse,
            -geometry.base.radial_shift_over_c / geometry.base.lapse,
            0.0,
            0.0,
        ),
        dtype=float,
    )
    radial = np.asarray(
        (0.0, 1.0 / np.sqrt(geometry.base.gamma_rr), 0.0, 0.0),
        dtype=float,
    )
    azimuthal = np.asarray((0.0, 0.0, 1.0 / geometry.radius, 0.0), dtype=float)
    vertical = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=float)

    gamma_phi = 1.0 / np.sqrt(1.0 - beta_phi**2)
    corotating_time = gamma_phi * (normal + beta_phi * azimuthal)
    rest_phi = gamma_phi * (beta_phi * normal + azimuthal)
    corotating_radial_speed = gamma_phi * beta_r
    if abs(corotating_radial_speed) >= 1.0:
        raise ValueError("composed horizontal velocity is not subluminal")
    gamma_r = 1.0 / np.sqrt(1.0 - corotating_radial_speed**2)
    horizontal_velocity = gamma_r * (
        corotating_time + corotating_radial_speed * radial
    )
    rest_radial = gamma_r * (
        corotating_radial_speed * corotating_time + radial
    )
    gamma_z = 1.0 / np.sqrt(1.0 - beta_z**2)
    four_velocity = gamma_z * (horizontal_velocity + beta_z * vertical)
    rest_vertical = gamma_z * (beta_z * horizontal_velocity + vertical)
    triad = np.asarray((rest_radial, rest_phi, rest_vertical), dtype=float)

    e_r, e_phi, e_z = triad
    basis = np.asarray(
        (
            (np.outer(e_r, e_r) - np.outer(e_phi, e_phi)) / np.sqrt(2.0),
            (
                np.outer(e_r, e_r)
                + np.outer(e_phi, e_phi)
                - 2.0 * np.outer(e_z, e_z)
            )
            / np.sqrt(6.0),
            _outer_symmetric(e_r, e_phi) / np.sqrt(2.0),
            _outer_symmetric(e_r, e_z) / np.sqrt(2.0),
            _outer_symmetric(e_phi, e_z) / np.sqrt(2.0),
        ),
        dtype=float,
    )
    return FullShearRestFrame(
        metric=metric,
        inverse_metric=inverse_metric,
        four_velocity=four_velocity,
        rest_triad=triad,
        stf_basis=basis,
    )


def reconstruct_full_shear_tensor(
    frame: FullShearRestFrame,
    amplitudes,
    *,
    stress_scale: float = 1.0,
) -> np.ndarray:
    values = _require_vector(amplitudes, N_FULL_SHEAR_AMPLITUDES, "amplitudes")
    scale = float(stress_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("stress scale must be positive and finite")
    return scale * np.einsum("a,aij->ij", values, frame.stf_basis)


def project_full_shear_amplitudes(
    frame: FullShearRestFrame,
    tensor,
    *,
    stress_scale: float = 1.0,
) -> np.ndarray:
    values = np.asarray(tensor, dtype=float)
    if values.shape != (4, 4) or np.any(~np.isfinite(values)):
        raise ValueError("full shear tensor must be finite and 4x4")
    scale = float(stress_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("stress scale must be positive and finite")
    lowered = np.einsum("ik,akl,lj->aij", frame.metric, frame.stf_basis, frame.metric)
    return np.einsum("aij,ij->a", lowered, values) / scale


def one_Rphi_amplitude_embedding(specific_stress: float) -> np.ndarray:
    """Embed the old coefficient multiplying ``e_R e_phi + e_phi e_R``."""

    value = float(specific_stress)
    if not np.isfinite(value):
        raise ValueError("specific stress must be finite")
    amplitudes = np.zeros(N_FULL_SHEAR_AMPLITUDES, dtype=float)
    amplitudes[2] = np.sqrt(2.0) * value
    return amplitudes


def audit_full_shear_rest_frame(
    frame: FullShearRestFrame,
    *,
    test_amplitudes=(0.017, -0.011, 0.023, 0.007, -0.013),
    old_specific_stress: float = 0.019,
) -> FullShearRestFrameAudit:
    metric = frame.metric
    lower_velocity = metric @ frame.four_velocity
    triad_gram = frame.rest_triad @ metric @ frame.rest_triad.T
    lowered_basis = np.einsum(
        "ik,akl,lj->aij", metric, frame.stf_basis, metric
    )
    basis_gram = np.einsum("aij,bij->ab", lowered_basis, frame.stf_basis)
    traces = np.einsum("ij,aij->a", metric, frame.stf_basis)
    velocity_contraction = np.einsum(
        "i,aij->aj", lower_velocity, frame.stf_basis
    )
    amplitudes = _require_vector(
        test_amplitudes, N_FULL_SHEAR_AMPLITUDES, "test amplitudes"
    )
    tensor = reconstruct_full_shear_tensor(frame, amplitudes, stress_scale=2.7)
    recovered = project_full_shear_amplitudes(frame, tensor, stress_scale=2.7)
    embedded = reconstruct_full_shear_tensor(
        frame,
        one_Rphi_amplitude_embedding(old_specific_stress),
    )
    expected = old_specific_stress * _outer_symmetric(
        frame.rest_triad[0], frame.rest_triad[1]
    )
    return FullShearRestFrameAudit(
        four_velocity_normalization_defect=abs(
            float(frame.four_velocity @ metric @ frame.four_velocity) + 1.0
        ),
        triad_orthonormality_defect=float(
            np.max(np.abs(triad_gram - np.eye(3)))
        ),
        triad_velocity_orthogonality_defect=float(
            np.max(np.abs(frame.rest_triad @ lower_velocity))
        ),
        basis_symmetry_defect=float(
            np.max(np.abs(frame.stf_basis - frame.stf_basis.swapaxes(1, 2)))
        ),
        basis_trace_defect=float(np.max(np.abs(traces))),
        basis_velocity_orthogonality_defect=float(
            np.max(np.abs(velocity_contraction))
        ),
        basis_gram_defect=float(
            np.max(np.abs(basis_gram - np.eye(N_FULL_SHEAR_AMPLITUDES)))
        ),
        amplitude_roundtrip_defect=float(np.max(np.abs(recovered - amplitudes))),
        one_Rphi_embedding_defect=float(np.max(np.abs(embedded - expected))),
    )


def reference_eleven_field_parameters() -> ElevenFieldConvexNormalFormParameters:
    """Return a deterministic structural fixture, not a physical calibration."""

    return ElevenFieldConvexNormalFormParameters(
        transport_speed_over_c=0.055,
        surface_density_weight=1.4,
        radial_velocity_weight=2.0,
        azimuthal_velocity_weight=1.8,
        thermal_weight=1.1,
        vertical_velocity_weight=1.3,
        shear_weights=(0.92, 0.96, 0.90, 0.94, 0.98),
        vertical_frequency=0.8,
        vertical_damping=0.12,
        shear_relaxation_rates=(0.35, 0.34, 0.36, 0.33, 0.37),
        height_log_surface_density=-0.16,
        height_log_temperature=1.24,
        mass_radial_coupling=0.25,
        radial_thermal_coupling=0.15,
        radial_height_coupling=0.07,
        thermal_height_coupling=0.05,
        radial_shear_diagonal_couplings=(0.075, 0.045),
        azimuthal_Rphi_coupling=0.16,
        vertical_Rz_coupling=0.13,
    )


def _chart_to_relaxation(parameters: ElevenFieldConvexNormalFormParameters) -> np.ndarray:
    matrix = np.eye(N_ELEVEN_FIELDS, dtype=float)
    matrix[4] = 0.0
    matrix[4, 0] = -parameters.height_log_surface_density
    matrix[4, 3] = -parameters.height_log_temperature
    matrix[4, 4] = 1.0
    return matrix


def _vertical_equilibrium_embedding(
    parameters: ElevenFieldConvexNormalFormParameters,
) -> np.ndarray:
    embedding = np.zeros(
        (N_ELEVEN_FIELDS, N_VERTICAL_EQUILIBRIUM_FIELDS), dtype=float
    )
    embedding[:4, :4] = np.eye(4)
    embedding[4, 0] = parameters.height_log_surface_density
    embedding[4, 3] = parameters.height_log_temperature
    embedding[6:, 4:] = np.eye(N_FULL_SHEAR_AMPLITUDES)
    return embedding


def build_eleven_field_convex_normal_form(
    parameters: ElevenFieldConvexNormalFormParameters,
) -> ElevenFieldConvexNormalForm:
    if not isinstance(parameters, ElevenFieldConvexNormalFormParameters):
        raise TypeError("parameters must be ElevenFieldConvexNormalFormParameters")
    transform = _chart_to_relaxation(parameters)
    height_weight = (
        parameters.vertical_velocity_weight * parameters.vertical_frequency**2
    )
    metric = np.diag(
        (
            parameters.surface_density_weight,
            parameters.radial_velocity_weight,
            parameters.azimuthal_velocity_weight,
            parameters.thermal_weight,
            height_weight,
            parameters.vertical_velocity_weight,
            *parameters.shear_weights,
        )
    )
    flux = parameters.transport_speed_over_c * metric
    couplings = (
        (0, 1, parameters.mass_radial_coupling),
        (1, 3, parameters.radial_thermal_coupling),
        (1, 4, parameters.radial_height_coupling),
        (3, 4, parameters.thermal_height_coupling),
        (1, 6, parameters.radial_shear_diagonal_couplings[0]),
        (1, 7, parameters.radial_shear_diagonal_couplings[1]),
        (2, 8, parameters.azimuthal_Rphi_coupling),
        (5, 9, parameters.vertical_Rz_coupling),
    )
    for first, second, value in couplings:
        flux[first, second] += value
        flux[second, first] += value

    generator = np.zeros((N_ELEVEN_FIELDS, N_ELEVEN_FIELDS), dtype=float)
    generator[4, 5] = 1.0
    generator[5, 4] = -(parameters.vertical_frequency**2)
    generator[5, 5] = -parameters.vertical_damping
    generator[6:, 6:] = -np.diag(parameters.shear_relaxation_rates)

    temporal = transform.T @ metric @ transform
    radial = transform.T @ flux @ transform
    source = transform.T @ metric @ generator @ transform
    embedding = _vertical_equilibrium_embedding(parameters)
    return ElevenFieldConvexNormalForm(
        chart_to_relaxation=transform,
        entropy_metric_relaxation=metric,
        flux_hessian_relaxation=flux,
        source_generator_relaxation=generator,
        temporal_matrix=temporal,
        radial_matrix=radial,
        source_matrix=source,
        vertical_equilibrium_embedding=embedding,
        reduced_temporal_matrix=embedding.T @ temporal @ embedding,
        reduced_radial_matrix=embedding.T @ radial @ embedding,
    )


def _generalized_symmetric_eigensystem(
    temporal: np.ndarray, radial: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    values, vectors = np.linalg.eigh(temporal)
    if float(np.min(values)) <= 0.0:
        raise ValueError("temporal Hessian must be positive definite")
    inverse_sqrt = vectors @ np.diag(values**-0.5) @ vectors.T
    symmetric = inverse_sqrt @ radial @ inverse_sqrt
    speeds, orthogonal = np.linalg.eigh(0.5 * (symmetric + symmetric.T))
    eigenvectors = inverse_sqrt @ orthogonal
    residual = radial @ eigenvectors - temporal @ (
        eigenvectors * speeds[None, :]
    )
    scale = max(
        float(np.linalg.norm(radial @ eigenvectors)),
        float(np.linalg.norm(temporal @ (eigenvectors * speeds[None, :]))),
        1.0,
    )
    return (
        speeds,
        eigenvectors,
        float(np.linalg.norm(residual) / scale),
        float(
            np.linalg.norm(
                eigenvectors.T @ temporal @ eigenvectors - np.eye(len(speeds))
            )
        ),
    )


def audit_eleven_field_convex_normal_form(
    parameters: ElevenFieldConvexNormalFormParameters,
) -> ElevenFieldConvexNormalFormAudit:
    form = build_eleven_field_convex_normal_form(parameters)
    temporal = form.temporal_matrix
    radial = form.radial_matrix
    speeds, _vectors, eigen_defect, orthogonality = (
        _generalized_symmetric_eigensystem(temporal, radial)
    )
    reduced_speeds, _vectors, reduced_defect, _orthogonality = (
        _generalized_symmetric_eigensystem(
            form.reduced_temporal_matrix, form.reduced_radial_matrix
        )
    )
    source_symmetric = 0.5 * (form.source_matrix + form.source_matrix.T)
    source_maximum = float(np.max(np.linalg.eigvalsh(source_symmetric)))
    transformed_embedding = (
        form.chart_to_relaxation @ form.vertical_equilibrium_embedding
    )
    expected_embedding = np.zeros_like(transformed_embedding)
    expected_embedding[:4, :4] = np.eye(4)
    expected_embedding[6:, 4:] = np.eye(N_FULL_SHEAR_AMPLITUDES)
    vertical_exchange = (
        form.entropy_metric_relaxation[4:6, 4:6]
        @ form.source_generator_relaxation[4:6, 4:6]
    )
    conservative_vertical = vertical_exchange.copy()
    conservative_vertical[1, 1] = 0.0
    chart = np.linspace(-0.13, 0.17, N_ELEVEN_FIELDS)
    state_gradient = form.state_current(chart)
    radial_gradient = form.radial_current(chart)
    lower = np.maximum(speeds[: len(reduced_speeds)] - reduced_speeds, 0.0)
    upper = np.maximum(reduced_speeds - speeds[2:], 0.0)
    return ElevenFieldConvexNormalFormAudit(
        temporal_minimum_eigenvalue=float(np.min(np.linalg.eigvalsh(temporal))),
        temporal_condition_number=float(np.linalg.cond(temporal)),
        temporal_symmetry_defect=float(
            np.linalg.norm(temporal - temporal.T) / max(np.linalg.norm(temporal), 1.0)
        ),
        radial_symmetry_defect=float(
            np.linalg.norm(radial - radial.T) / max(np.linalg.norm(radial), 1.0)
        ),
        state_gradient_defect=float(
            np.linalg.norm(state_gradient - temporal @ chart)
        ),
        radial_gradient_defect=float(
            np.linalg.norm(radial_gradient - radial @ chart)
        ),
        generalized_eigenpair_defect=eigen_defect,
        energy_metric_orthogonality_defect=orthogonality,
        maximum_absolute_characteristic_speed_over_c=float(np.max(np.abs(speeds))),
        source_entropy_positive_part=max(source_maximum, 0.0),
        vertical_hamiltonian_entropy_defect=float(
            np.linalg.norm(conservative_vertical + conservative_vertical.T)
        ),
        vertical_equilibrium_embedding_defect=float(
            np.linalg.norm(transformed_embedding - expected_embedding)
        ),
        reduced_temporal_minimum_eigenvalue=float(
            np.min(np.linalg.eigvalsh(form.reduced_temporal_matrix))
        ),
        reduced_generalized_eigenpair_defect=reduced_defect,
        subcharacteristic_interlacing_violation=float(
            max(np.max(lower), np.max(upper))
        ),
        full_characteristic_speeds_over_c=tuple(float(value) for value in speeds),
        reduced_characteristic_speeds_over_c=tuple(
            float(value) for value in reduced_speeds
        ),
    )


__all__ = [
    "ELEVEN_FIELD_PERTURBATION_NAMES",
    "FULL_SHEAR_AMPLITUDE_NAMES",
    "N_ELEVEN_FIELDS",
    "N_FULL_SHEAR_AMPLITUDES",
    "N_VERTICAL_EQUILIBRIUM_FIELDS",
    "VERTICAL_EQUILIBRIUM_NINE_FIELD_NAMES",
    "ElevenFieldConvexNormalForm",
    "ElevenFieldConvexNormalFormAudit",
    "ElevenFieldConvexNormalFormParameters",
    "FullShearRestFrame",
    "FullShearRestFrameAudit",
    "audit_eleven_field_convex_normal_form",
    "audit_full_shear_rest_frame",
    "build_eleven_field_convex_normal_form",
    "full_shear_rest_frame",
    "one_Rphi_amplitude_embedding",
    "project_full_shear_amplitudes",
    "reconstruct_full_shear_tensor",
    "reference_eleven_field_parameters",
]
