"""Entropy-variable normal form for the proposed seven-field inner flow.

This module is a mathematical design aid.  It does not replace the certified
five-field implementation and it does not advance a trajectory.

The proposed primitive perturbations are ordered as

``(ln Sigma, beta_r, beta_phi, ln T, chi, ln H, w_H/c)``.

The height departure from the local quasi-hydrostatic manifold is

``y_H = dlnH - H_Sigma dlnSigma - H_T dlnT``.

For an invertible chart map ``r = L q``, a positive diagonal entropy metric
``M`` and a symmetric flux Hessian ``K``, the chart matrices

``A0 = L.T M L`` and ``A1 = L.T K L``

are symmetric with ``A0`` positive definite.  The generalized principal
pencil is therefore self-adjoint in the ``A0`` metric.  The vertical
oscillator and stress relaxation source is chosen so its entropy production
is non-positive.  Restriction to ``y_H = w_H = 0`` is a five-dimensional
compression, so its characteristic speeds interlace the seven-field speeds.

The construction is the Stage-1 proof template.  A physical Kerr--Schild
model must derive its nonlinear entropy and flux potentials and calibrate the
coefficients independently before this normal form can become production
code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SEVEN_FIELD_PRIMITIVE_NAMES = (
    "log_surface_density",
    "radial_velocity_over_c",
    "azimuthal_velocity_over_c",
    "log_temperature",
    "specific_shear_stress",
    "log_height",
    "vertical_velocity_over_c",
)

VERTICAL_EQUILIBRIUM_FIVE_FIELD_NAMES = (
    "log_surface_density",
    "radial_velocity_over_c",
    "azimuthal_velocity_over_c",
    "log_temperature",
    "specific_shear_stress",
)

N_SEVEN_FIELDS = len(SEVEN_FIELD_PRIMITIVE_NAMES)
N_VERTICAL_EQUILIBRIUM_FIELDS = len(
    VERTICAL_EQUILIBRIUM_FIVE_FIELD_NAMES
)


@dataclass(frozen=True)
class SevenFieldEntropyNormalFormParameters:
    """Positive metric data and symmetric principal couplings.

    All entries are nondimensional local coefficients.  Stage 2 must obtain
    their physical counterparts from the relativistic column energy and flux
    potentials rather than fitting them to the failed characteristic face.
    """

    transport_speed_over_c: float
    surface_density_weight: float
    radial_velocity_weight: float
    azimuthal_velocity_weight: float
    thermal_weight: float
    stress_weight: float
    vertical_velocity_weight: float
    vertical_frequency: float
    vertical_damping: float
    stress_relaxation_rate: float
    height_log_surface_density: float
    height_log_temperature: float
    mass_radial_coupling: float
    radial_thermal_coupling: float
    radial_height_coupling: float
    thermal_height_coupling: float
    azimuthal_stress_coupling: float

    def __post_init__(self) -> None:
        finite = tuple(float(value) for value in self.__dict__.values())
        if any(not np.isfinite(value) for value in finite):
            raise ValueError("seven-field normal-form coefficients must be finite")
        positive = (
            self.surface_density_weight,
            self.radial_velocity_weight,
            self.azimuthal_velocity_weight,
            self.thermal_weight,
            self.stress_weight,
            self.vertical_velocity_weight,
            self.vertical_frequency,
            self.stress_relaxation_rate,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("entropy weights and relaxation scales must be positive")
        if self.vertical_damping < 0.0:
            raise ValueError("vertical damping must be non-negative")


@dataclass(frozen=True)
class SevenFieldEntropyNormalForm:
    """Matrices of the entropy-variable Stage-1 normal form."""

    chart_to_relaxation: np.ndarray
    entropy_metric_relaxation: np.ndarray
    flux_hessian_relaxation: np.ndarray
    source_generator_relaxation: np.ndarray
    temporal_matrix: np.ndarray
    spatial_matrix: np.ndarray
    source_matrix: np.ndarray
    vertical_equilibrium_embedding: np.ndarray
    reduced_temporal_matrix: np.ndarray
    reduced_spatial_matrix: np.ndarray


@dataclass(frozen=True)
class SevenFieldEntropyNormalFormAudit:
    """Numerical audit of identities that are analytic by construction."""

    temporal_minimum_eigenvalue: float
    temporal_condition_number: float
    temporal_symmetry_defect: float
    spatial_symmetry_defect: float
    generalized_eigenpair_defect: float
    energy_metric_orthogonality_defect: float
    maximum_characteristic_imaginary_part: float
    maximum_absolute_characteristic_speed_over_c: float
    source_entropy_maximum_eigenvalue: float
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
            and self.temporal_symmetry_defect <= 1.0e-12
            and self.spatial_symmetry_defect <= 1.0e-12
            and self.generalized_eigenpair_defect <= 1.0e-11
            and self.energy_metric_orthogonality_defect <= 1.0e-11
            and self.maximum_characteristic_imaginary_part <= 1.0e-14
            and self.maximum_absolute_characteristic_speed_over_c < 1.0
            and self.source_entropy_positive_part <= 1.0e-12
            and self.vertical_hamiltonian_entropy_defect <= 1.0e-12
            and self.vertical_equilibrium_embedding_defect <= 1.0e-12
            and self.reduced_generalized_eigenpair_defect <= 1.0e-11
            and self.subcharacteristic_interlacing_violation <= 1.0e-12
        )


def reference_seven_field_entropy_parameters(
    ) -> SevenFieldEntropyNormalFormParameters:
    """Return a deterministic, causal Stage-1 identity-test fixture."""

    return SevenFieldEntropyNormalFormParameters(
        transport_speed_over_c=0.07,
        surface_density_weight=1.40,
        radial_velocity_weight=2.00,
        azimuthal_velocity_weight=1.80,
        thermal_weight=1.10,
        stress_weight=0.90,
        vertical_velocity_weight=1.30,
        vertical_frequency=0.80,
        vertical_damping=0.12,
        stress_relaxation_rate=0.35,
        height_log_surface_density=-0.16,
        height_log_temperature=1.24,
        mass_radial_coupling=0.31,
        radial_thermal_coupling=0.19,
        radial_height_coupling=0.10,
        thermal_height_coupling=0.06,
        azimuthal_stress_coupling=0.27,
    )


def chart_to_relaxation_matrix(
    *,
    height_log_surface_density: float,
    height_log_temperature: float,
) -> np.ndarray:
    """Return ``L`` with height departure as its sixth coordinate."""

    h_sigma = float(height_log_surface_density)
    h_temperature = float(height_log_temperature)
    if not np.isfinite(h_sigma) or not np.isfinite(h_temperature):
        raise ValueError("height response coefficients must be finite")
    matrix = np.eye(N_SEVEN_FIELDS, dtype=float)
    matrix[5] = 0.0
    matrix[5, 0] = -h_sigma
    matrix[5, 3] = -h_temperature
    matrix[5, 5] = 1.0
    return matrix


def vertical_equilibrium_embedding(
    *,
    height_log_surface_density: float,
    height_log_temperature: float,
) -> np.ndarray:
    """Embed five fields in seven with hydrostatic height and zero velocity."""

    h_sigma = float(height_log_surface_density)
    h_temperature = float(height_log_temperature)
    if not np.isfinite(h_sigma) or not np.isfinite(h_temperature):
        raise ValueError("height response coefficients must be finite")
    embedding = np.zeros(
        (N_SEVEN_FIELDS, N_VERTICAL_EQUILIBRIUM_FIELDS),
        dtype=float,
    )
    embedding[:N_VERTICAL_EQUILIBRIUM_FIELDS] = np.eye(
        N_VERTICAL_EQUILIBRIUM_FIELDS,
        dtype=float,
    )
    embedding[5, 0] = h_sigma
    embedding[5, 3] = h_temperature
    return embedding


def build_seven_field_entropy_normal_form(
    parameters: SevenFieldEntropyNormalFormParameters,
) -> SevenFieldEntropyNormalForm:
    """Build the symmetric principal matrices and dissipative source."""

    if not isinstance(parameters, SevenFieldEntropyNormalFormParameters):
        raise TypeError("parameters must be SevenFieldEntropyNormalFormParameters")

    transform = chart_to_relaxation_matrix(
        height_log_surface_density=parameters.height_log_surface_density,
        height_log_temperature=parameters.height_log_temperature,
    )
    omega_squared = parameters.vertical_frequency**2
    height_weight = parameters.vertical_velocity_weight * omega_squared
    metric = np.diag(
        (
            parameters.surface_density_weight,
            parameters.radial_velocity_weight,
            parameters.azimuthal_velocity_weight,
            parameters.thermal_weight,
            parameters.stress_weight,
            height_weight,
            parameters.vertical_velocity_weight,
        )
    )

    flux = parameters.transport_speed_over_c * metric
    couplings = (
        (0, 1, parameters.mass_radial_coupling),
        (1, 3, parameters.radial_thermal_coupling),
        (1, 5, parameters.radial_height_coupling),
        (3, 5, parameters.thermal_height_coupling),
        (2, 4, parameters.azimuthal_stress_coupling),
    )
    for first, second, value in couplings:
        flux[first, second] += value
        flux[second, first] += value

    generator = np.zeros((N_SEVEN_FIELDS, N_SEVEN_FIELDS), dtype=float)
    generator[4, 4] = -parameters.stress_relaxation_rate
    generator[5, 6] = 1.0
    generator[6, 5] = -omega_squared
    generator[6, 6] = -parameters.vertical_damping

    temporal = transform.T @ metric @ transform
    spatial = transform.T @ flux @ transform
    source = transform.T @ metric @ generator @ transform
    embedding = vertical_equilibrium_embedding(
        height_log_surface_density=parameters.height_log_surface_density,
        height_log_temperature=parameters.height_log_temperature,
    )
    reduced_temporal = embedding.T @ temporal @ embedding
    reduced_spatial = embedding.T @ spatial @ embedding
    return SevenFieldEntropyNormalForm(
        chart_to_relaxation=transform,
        entropy_metric_relaxation=metric,
        flux_hessian_relaxation=flux,
        source_generator_relaxation=generator,
        temporal_matrix=temporal,
        spatial_matrix=spatial,
        source_matrix=source,
        vertical_equilibrium_embedding=embedding,
        reduced_temporal_matrix=reduced_temporal,
        reduced_spatial_matrix=reduced_spatial,
    )


def _generalized_symmetric_eigensystem(
    temporal: np.ndarray,
    spatial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return real speeds, energy-orthonormal vectors and audit defects."""

    temporal_values, temporal_vectors = np.linalg.eigh(temporal)
    if float(np.min(temporal_values)) <= 0.0:
        raise ValueError("temporal entropy matrix must be positive definite")
    inverse_square_root = (
        temporal_vectors
        @ np.diag(temporal_values**-0.5)
        @ temporal_vectors.T
    )
    symmetric_operator = inverse_square_root @ spatial @ inverse_square_root
    symmetric_operator = 0.5 * (symmetric_operator + symmetric_operator.T)
    speeds, orthogonal_vectors = np.linalg.eigh(symmetric_operator)
    vectors = inverse_square_root @ orthogonal_vectors
    residual = spatial @ vectors - temporal @ (
        vectors * speeds[None, :]
    )
    residual_scale = max(
        float(np.linalg.norm(spatial @ vectors)),
        float(np.linalg.norm(temporal @ (vectors * speeds[None, :]))),
        1.0,
    )
    eigenpair_defect = float(np.linalg.norm(residual) / residual_scale)
    metric_orthogonality = float(
        np.linalg.norm(vectors.T @ temporal @ vectors - np.eye(len(speeds)))
    )
    return speeds, vectors, eigenpair_defect, metric_orthogonality


def audit_seven_field_entropy_normal_form(
    parameters: SevenFieldEntropyNormalFormParameters,
) -> SevenFieldEntropyNormalFormAudit:
    """Audit positivity, symmetry, entropy dissipation and interlacing."""

    normal_form = build_seven_field_entropy_normal_form(parameters)
    temporal = normal_form.temporal_matrix
    spatial = normal_form.spatial_matrix
    temporal_values = np.linalg.eigvalsh(temporal)
    speeds, _vectors, eigenpair_defect, orthogonality = (
        _generalized_symmetric_eigensystem(temporal, spatial)
    )
    reduced_speeds, _reduced_vectors, reduced_defect, _ = (
        _generalized_symmetric_eigensystem(
            normal_form.reduced_temporal_matrix,
            normal_form.reduced_spatial_matrix,
        )
    )

    source_entropy = 0.5 * (
        normal_form.source_matrix + normal_form.source_matrix.T
    )
    source_entropy_values = np.linalg.eigvalsh(source_entropy)
    source_maximum = float(np.max(source_entropy_values))
    expected_embedding = np.zeros(
        (N_SEVEN_FIELDS, N_VERTICAL_EQUILIBRIUM_FIELDS),
        dtype=float,
    )
    expected_embedding[:N_VERTICAL_EQUILIBRIUM_FIELDS] = np.eye(
        N_VERTICAL_EQUILIBRIUM_FIELDS,
    )
    transformed_embedding = (
        normal_form.chart_to_relaxation
        @ normal_form.vertical_equilibrium_embedding
    )

    metric = normal_form.entropy_metric_relaxation
    generator = normal_form.source_generator_relaxation
    vertical_exchange = metric[5:7, 5:7] @ generator[5:7, 5:7]
    vertical_conservative_part = vertical_exchange.copy()
    vertical_conservative_part[1, 1] = 0.0
    vertical_hamiltonian_defect = float(
        np.linalg.norm(
            vertical_conservative_part + vertical_conservative_part.T
        )
    )

    lower_violation = np.maximum(speeds[: len(reduced_speeds)] - reduced_speeds, 0.0)
    upper_violation = np.maximum(reduced_speeds - speeds[2:], 0.0)
    interlacing_violation = float(
        max(np.max(lower_violation), np.max(upper_violation))
    )

    temporal_scale = max(float(np.linalg.norm(temporal)), 1.0)
    spatial_scale = max(float(np.linalg.norm(spatial)), 1.0)
    return SevenFieldEntropyNormalFormAudit(
        temporal_minimum_eigenvalue=float(np.min(temporal_values)),
        temporal_condition_number=float(np.linalg.cond(temporal)),
        temporal_symmetry_defect=float(
            np.linalg.norm(temporal - temporal.T) / temporal_scale
        ),
        spatial_symmetry_defect=float(
            np.linalg.norm(spatial - spatial.T) / spatial_scale
        ),
        generalized_eigenpair_defect=eigenpair_defect,
        energy_metric_orthogonality_defect=orthogonality,
        maximum_characteristic_imaginary_part=0.0,
        maximum_absolute_characteristic_speed_over_c=float(
            np.max(np.abs(speeds))
        ),
        source_entropy_maximum_eigenvalue=source_maximum,
        source_entropy_positive_part=max(source_maximum, 0.0),
        vertical_hamiltonian_entropy_defect=vertical_hamiltonian_defect,
        vertical_equilibrium_embedding_defect=float(
            np.linalg.norm(transformed_embedding - expected_embedding)
        ),
        reduced_temporal_minimum_eigenvalue=float(
            np.min(np.linalg.eigvalsh(normal_form.reduced_temporal_matrix))
        ),
        reduced_generalized_eigenpair_defect=reduced_defect,
        subcharacteristic_interlacing_violation=interlacing_violation,
        full_characteristic_speeds_over_c=tuple(float(value) for value in speeds),
        reduced_characteristic_speeds_over_c=tuple(
            float(value) for value in reduced_speeds
        ),
    )


__all__ = [
    "N_SEVEN_FIELDS",
    "N_VERTICAL_EQUILIBRIUM_FIELDS",
    "SEVEN_FIELD_PRIMITIVE_NAMES",
    "VERTICAL_EQUILIBRIUM_FIVE_FIELD_NAMES",
    "SevenFieldEntropyNormalForm",
    "SevenFieldEntropyNormalFormAudit",
    "SevenFieldEntropyNormalFormParameters",
    "audit_seven_field_entropy_normal_form",
    "build_seven_field_entropy_normal_form",
    "chart_to_relaxation_matrix",
    "reference_seven_field_entropy_parameters",
    "vertical_equilibrium_embedding",
]
