"""Proof kernel for the split Godunov/port-Hamiltonian architecture."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from imri_qpe.constants import C, DEFAULT_MU_MOL
from imri_qpe.scales import gas_constant_per_gram

from .causal_inner_eleven_field_convex import (
    build_eleven_field_convex_normal_form,
    reference_eleven_field_parameters,
)


SPLIT_ELEVEN_FIELD_NAMES = (
    "log_surface_density",
    "radial_velocity_over_c",
    "azimuthal_velocity_over_c",
    "log_temperature",
    "zeta_RR_minus_phiphi",
    "zeta_RR_plus_phiphi_minus_2zz",
    "zeta_Rphi",
    "zeta_Rz",
    "zeta_phiz",
    "log_height_over_anchor",
    "vertical_velocity_over_c",
)


@dataclass(frozen=True)
class VerticalPortHamiltonian:
    """Positive metric, skew port and dissipative vertical generator."""

    effective_vertical_frequency_squared: float
    height_to_velocity_rate: float
    velocity_to_height_rate: float
    damping_rate: float
    entropy_metric: np.ndarray
    rate_matrix: np.ndarray
    source_matrix: np.ndarray
    reversible_source_matrix: np.ndarray
    radial_matrix: np.ndarray


@dataclass(frozen=True)
class SplitGodunovPortHamiltonianForm:
    """Combined nine-field transport and two-field vertical proof form."""

    temporal_matrix: np.ndarray
    radial_matrix: np.ndarray
    source_matrix: np.ndarray
    transport_temporal_matrix: np.ndarray
    transport_radial_matrix: np.ndarray
    transport_source_matrix: np.ndarray
    vertical_port: VerticalPortHamiltonian


@dataclass(frozen=True)
class SplitGodunovPortHamiltonianAudit:
    """Binding algebraic defects for one physical vertical witness."""

    equilibrated_temporal_minimum_eigenvalue: float
    equilibrated_temporal_condition_number: float
    temporal_symmetry_defect: float
    radial_symmetry_defect: float
    generalized_eigenpair_defect: float
    maximum_absolute_characteristic_speed_over_c: float
    port_skew_relative_defect: float
    source_entropy_positive_part: float
    vertical_reversible_energy_relative_defect: float
    vertical_damping_heat_ledger_relative_defect: float
    effective_vertical_frequency_squared: float
    characteristic_speeds_over_c: tuple[float, ...]

    @property
    def passed(self) -> bool:
        return (
            self.equilibrated_temporal_minimum_eigenvalue >= 1.0e-10
            and self.temporal_symmetry_defect <= 1.0e-12
            and self.radial_symmetry_defect <= 1.0e-12
            and self.generalized_eigenpair_defect <= 1.0e-11
            and self.maximum_absolute_characteristic_speed_over_c <= 0.999
            and self.port_skew_relative_defect <= 1.0e-12
            and self.source_entropy_positive_part <= 1.0e-12
            and self.vertical_reversible_energy_relative_defect <= 1.0e-12
            and self.vertical_damping_heat_ledger_relative_defect <= 1.0e-12
            and self.effective_vertical_frequency_squared > 0.0
        )


def vertical_port_hamiltonian(
    *,
    proper_half_thickness: float,
    temperature: float,
    proper_vertical_frequency: float,
    alpha: float,
    transport_speed_over_c: float,
    mu_mol: float = DEFAULT_MU_MOL,
) -> VerticalPortHamiltonian:
    """Return the physical local linear vertical port at one anchor."""

    height = float(proper_half_thickness)
    temp = float(temperature)
    omega = float(proper_vertical_frequency)
    alpha_value = float(alpha)
    transport = float(transport_speed_over_c)
    if min(height, temp, omega, alpha_value) <= 0.0:
        raise ValueError("vertical port inputs must be positive")
    if alpha_value >= 1.0 or abs(transport) >= 1.0:
        raise ValueError("vertical port alpha and transport must be causal")
    gas_constant = gas_constant_per_gram(mu_mol)
    effective_squared = omega**2 + gas_constant * temp / height**2
    height_to_velocity = C / height
    velocity_to_height = height * effective_squared / C
    damping = alpha_value * omega
    metric = np.diag((velocity_to_height, height_to_velocity))
    rate = np.asarray(
        ((0.0, height_to_velocity), (-velocity_to_height, -damping)),
        dtype=float,
    )
    source = metric @ rate
    reversible = np.array(source, copy=True)
    reversible[1, 1] = 0.0
    return VerticalPortHamiltonian(
        effective_vertical_frequency_squared=float(effective_squared),
        height_to_velocity_rate=float(height_to_velocity),
        velocity_to_height_rate=float(velocity_to_height),
        damping_rate=float(damping),
        entropy_metric=metric,
        rate_matrix=rate,
        source_matrix=source,
        reversible_source_matrix=reversible,
        radial_matrix=transport * metric,
    )


def build_split_godunov_port_hamiltonian_form(
    *,
    proper_half_thickness: float,
    temperature: float,
    proper_vertical_frequency: float,
    alpha: float,
    transport_speed_over_c: float,
) -> SplitGodunovPortHamiltonianForm:
    """Build the block proof form with a frozen-height transport generator."""

    parameters = replace(
        reference_eleven_field_parameters(),
        transport_speed_over_c=float(transport_speed_over_c),
    )
    full = build_eleven_field_convex_normal_form(parameters)
    embedding = full.vertical_equilibrium_embedding
    transport_temporal = full.reduced_temporal_matrix
    transport_radial = full.reduced_radial_matrix
    transport_source = embedding.T @ full.source_matrix @ embedding
    vertical = vertical_port_hamiltonian(
        proper_half_thickness=proper_half_thickness,
        temperature=temperature,
        proper_vertical_frequency=proper_vertical_frequency,
        alpha=alpha,
        transport_speed_over_c=transport_speed_over_c,
    )
    temporal = np.zeros((11, 11), dtype=float)
    radial = np.zeros_like(temporal)
    source = np.zeros_like(temporal)
    temporal[:9, :9] = transport_temporal
    temporal[9:, 9:] = vertical.entropy_metric
    radial[:9, :9] = transport_radial
    radial[9:, 9:] = vertical.radial_matrix
    source[:9, :9] = transport_source
    source[9:, 9:] = vertical.source_matrix
    return SplitGodunovPortHamiltonianForm(
        temporal_matrix=temporal,
        radial_matrix=radial,
        source_matrix=source,
        transport_temporal_matrix=transport_temporal,
        transport_radial_matrix=transport_radial,
        transport_source_matrix=transport_source,
        vertical_port=vertical,
    )


def _equilibrate(temporal: np.ndarray, *matrices: np.ndarray):
    diagonal = np.diag(temporal)
    if np.any(diagonal <= 0.0):
        raise ValueError("split temporal diagonal must be positive")
    inverse = np.diag(diagonal**-0.5)
    return (inverse @ temporal @ inverse,) + tuple(
        inverse @ matrix @ inverse for matrix in matrices
    )


def audit_split_godunov_port_hamiltonian_form(
    form: SplitGodunovPortHamiltonianForm,
) -> SplitGodunovPortHamiltonianAudit:
    """Audit positivity, causality, skew exchange and damping heat."""

    temporal, radial, source = _equilibrate(
        form.temporal_matrix, form.radial_matrix, form.source_matrix
    )
    temporal = 0.5 * (temporal + temporal.T)
    values, vectors = np.linalg.eigh(temporal)
    inverse_sqrt = vectors @ np.diag(values**-0.5) @ vectors.T
    symmetric_pencil = inverse_sqrt @ radial @ inverse_sqrt
    speeds, orthogonal = np.linalg.eigh(
        0.5 * (symmetric_pencil + symmetric_pencil.T)
    )
    eigenvectors = inverse_sqrt @ orthogonal
    residual = radial @ eigenvectors - temporal @ (
        eigenvectors * speeds[None, :]
    )
    residual_scale = max(
        float(np.linalg.norm(radial @ eigenvectors)),
        float(np.linalg.norm(temporal @ (eigenvectors * speeds[None, :]))),
        1.0,
    )
    port = form.vertical_port
    reversible_scale = max(
        float(np.linalg.norm(port.reversible_source_matrix)), 1.0
    )
    port_skew = float(
        np.linalg.norm(
            port.reversible_source_matrix + port.reversible_source_matrix.T
        )
        / reversible_scale
    )
    source_symmetric = 0.5 * (source + source.T)
    source_positive = max(float(np.max(np.linalg.eigvalsh(source_symmetric))), 0.0)
    probe = np.asarray((0.17, -0.11))
    reversible_rate = float(probe @ port.reversible_source_matrix @ probe)
    reversible_defect = abs(reversible_rate) / max(
        float(np.linalg.norm(port.reversible_source_matrix) * np.dot(probe, probe)),
        np.finfo(float).tiny,
    )
    dissipative_rate = float(
        probe @ (port.source_matrix - port.reversible_source_matrix) @ probe
    )
    deposited_heat = -dissipative_rate
    ledger_defect = abs(dissipative_rate + deposited_heat) / max(
        abs(dissipative_rate), abs(deposited_heat), np.finfo(float).tiny
    )
    temporal_scale = max(float(np.linalg.norm(form.temporal_matrix)), 1.0)
    radial_scale = max(float(np.linalg.norm(form.radial_matrix)), 1.0)
    return SplitGodunovPortHamiltonianAudit(
        equilibrated_temporal_minimum_eigenvalue=float(np.min(values)),
        equilibrated_temporal_condition_number=float(np.linalg.cond(temporal)),
        temporal_symmetry_defect=float(
            np.linalg.norm(form.temporal_matrix - form.temporal_matrix.T)
            / temporal_scale
        ),
        radial_symmetry_defect=float(
            np.linalg.norm(form.radial_matrix - form.radial_matrix.T) / radial_scale
        ),
        generalized_eigenpair_defect=float(np.linalg.norm(residual) / residual_scale),
        maximum_absolute_characteristic_speed_over_c=float(np.max(np.abs(speeds))),
        port_skew_relative_defect=port_skew,
        source_entropy_positive_part=source_positive,
        vertical_reversible_energy_relative_defect=float(reversible_defect),
        vertical_damping_heat_ledger_relative_defect=float(ledger_defect),
        effective_vertical_frequency_squared=port.effective_vertical_frequency_squared,
        characteristic_speeds_over_c=tuple(float(value) for value in speeds),
    )


__all__ = [
    "SPLIT_ELEVEN_FIELD_NAMES",
    "SplitGodunovPortHamiltonianAudit",
    "SplitGodunovPortHamiltonianForm",
    "VerticalPortHamiltonian",
    "audit_split_godunov_port_hamiltonian_form",
    "build_split_godunov_port_hamiltonian_form",
    "vertical_port_hamiltonian",
]
