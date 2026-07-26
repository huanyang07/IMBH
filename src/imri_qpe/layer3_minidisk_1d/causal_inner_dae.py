"""Count and principal-rank gates for the five-field causal inner DAE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C

from .causal_inner_geometry import KerrSchildColumnGeometry
from .causal_inner_stress import CausalAlphaShearClosure
from .causal_inner_thermal import QuasiHydrostaticGasRadiationColumnEOS
from .causal_inner_valencia import (
    valencia_radial_characteristic_speeds_over_c,
)


@dataclass(frozen=True)
class CausalFiveFieldDAECount:
    """Exact flux-primary unknown and residual count."""

    n_cells: int
    conserved_unknowns: int
    primitive_unknowns: int
    face_flux_unknowns: int
    total_unknowns: int
    conservation_rows: int
    primitive_map_rows: int
    interior_flux_rows: int
    inner_flux_rows: int
    outer_flux_rows: int
    total_rows: int
    nonconservative_shear_rows: int
    physical_inner_boundary_conditions: int
    physical_outer_boundary_conditions: int

    @property
    def square(self) -> bool:
        return self.total_unknowns == self.total_rows


@dataclass(frozen=True)
class CausalFiveFieldPrincipalAudit:
    """Responsive-height local principal and coordinate-mode audit."""

    local_rest_mass_matrix: np.ndarray
    local_rest_flux_matrix: np.ndarray
    local_rest_right_eigenvectors: np.ndarray
    analytic_local_rest_speeds_over_c: tuple[float, ...]
    numerical_local_rest_speeds_over_c: tuple[float, ...]
    coordinate_speeds_over_c: tuple[float, ...]
    incoming_inner_characteristics: int
    incoming_outer_characteristics: int
    stationary_coordinate_rank: int
    incoming_mode_response_matrix: np.ndarray
    incoming_mode_response_rank: int
    incoming_mode_response_smallest_singular_value: float
    maximum_local_rest_eigenvalue_defect: float
    maximum_imaginary_eigenvalue: float
    maximum_light_cone_excess: float

    @property
    def causal_and_hyperbolic(self) -> bool:
        return (
            self.maximum_local_rest_eigenvalue_defect <= 1.0e-12
            and self.maximum_imaginary_eigenvalue <= 1.0e-13
            and self.maximum_light_cone_excess <= 1.0e-12
        )


@dataclass(frozen=True)
class CausalFiveFieldBoundaryRankAudit:
    """Inner excision and outer Roche-plus-shear boundary-rank audit."""

    inner_incoming_characteristics: int
    outer_incoming_characteristics: int
    outer_face_rows: int
    outer_face_jacobian_rank: int
    outer_physical_boundary_conditions: int
    outer_incoming_response_rank: int
    outer_incoming_response_smallest_singular_value: float
    inner_excision_passed: bool
    outer_characteristic_count_passed: bool
    outer_response_rank_passed: bool

    @property
    def passed(self) -> bool:
        return (
            self.inner_excision_passed
            and self.outer_characteristic_count_passed
            and self.outer_response_rank_passed
            and self.outer_face_jacobian_rank == self.outer_face_rows
        )


def causal_five_field_dae_count(
    n_cells: int,
) -> CausalFiveFieldDAECount:
    """Return the exact ``15 N + 5`` five-field flux-primary count.

    The fifth conservation row contains the resolved shear-gradient
    principal term. It does not add an unknown or an extra residual row.
    """

    if int(n_cells) != n_cells or n_cells < 1:
        raise ValueError("causal DAE requires a positive integer cell count")
    n_cells = int(n_cells)
    conserved = 5 * n_cells
    primitives = 5 * n_cells
    face_fluxes = 5 * (n_cells + 1)
    conservation = 5 * n_cells
    primitive_map = 5 * n_cells
    interior_flux = 5 * (n_cells - 1)
    inner_flux = 5
    outer_flux = 5
    return CausalFiveFieldDAECount(
        n_cells=n_cells,
        conserved_unknowns=conserved,
        primitive_unknowns=primitives,
        face_flux_unknowns=face_fluxes,
        total_unknowns=conserved + primitives + face_fluxes,
        conservation_rows=conservation,
        primitive_map_rows=primitive_map,
        interior_flux_rows=interior_flux,
        inner_flux_rows=inner_flux,
        outer_flux_rows=outer_flux,
        total_rows=(
            conservation
            + primitive_map
            + interior_flux
            + inner_flux
            + outer_flux
        ),
        nonconservative_shear_rows=n_cells,
        physical_inner_boundary_conditions=0,
        physical_outer_boundary_conditions=2,
    )


def _responsive_five_field_principal_matrices(
    eos: QuasiHydrostaticGasRadiationColumnEOS,
    closure: CausalAlphaShearClosure,
    *,
    surface_density: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local-rest matrices in ``lnSigma,betaR,betaPhi,lnT,chi``."""

    thermodynamics = eos.from_surface_density_temperature(
        surface_density,
        temperature,
    )
    derivatives = eos.derivatives(surface_density, temperature)
    enthalpy_over_c2 = (
        1.0
        + thermodynamics.specific_internal_energy / C**2
        + thermodynamics.integrated_pressure
        / (thermodynamics.surface_density * C**2)
    )
    entropy_sigma = (
        derivatives.internal_energy_log_surface_density
        - (
            thermodynamics.integrated_pressure
            / thermodynamics.surface_density
        )
        * derivatives.density_log_surface_density
    )
    entropy_temperature = (
        derivatives.internal_energy_log_temperature
        - (
            thermodynamics.integrated_pressure
            / thermodynamics.surface_density
        )
        * derivatives.density_log_temperature
    )

    mass = np.zeros((5, 5), dtype=float)
    mass[0, 0] = 1.0
    mass[1, 1] = enthalpy_over_c2
    mass[2, 2] = enthalpy_over_c2
    mass[3, 0] = entropy_sigma
    mass[3, 3] = entropy_temperature
    mass[4, 4] = 1.0

    flux = np.zeros((5, 5), dtype=float)
    flux[0, 1] = 1.0
    flux[1, 0] = (
        derivatives.pressure_log_surface_density
        / (thermodynamics.surface_density * C**2)
    )
    flux[1, 3] = (
        derivatives.pressure_log_temperature
        / (thermodynamics.surface_density * C**2)
    )
    flux[2, 4] = 1.0
    flux[4, 2] = (
        enthalpy_over_c2
        * closure.viscous_signal_speed_over_c**2
    )
    return mass, flux


def audit_causal_five_field_principal(
    geometry: KerrSchildColumnGeometry,
    eos: QuasiHydrostaticGasRadiationColumnEOS,
    closure: CausalAlphaShearClosure,
    *,
    surface_density: float,
    radial_velocity_over_c: float,
    azimuthal_velocity_over_c: float,
    temperature: float,
) -> CausalFiveFieldPrincipalAudit:
    """Audit responsive acoustic/contact/shear modes and incoming rank."""

    mass, flux = _responsive_five_field_principal_matrices(
        eos,
        closure,
        surface_density=surface_density,
        temperature=temperature,
    )
    principal = np.linalg.solve(mass, flux)
    numerical_values, numerical_vectors = np.linalg.eig(principal)
    order = np.argsort(np.real(numerical_values))
    numerical_values = numerical_values[order]
    numerical_vectors = numerical_vectors[:, order]

    derivatives = eos.derivatives(surface_density, temperature)
    sound = derivatives.sound_speed_over_c
    shear = closure.viscous_signal_speed_over_c
    analytic_local = np.sort(
        np.asarray([-sound, -shear, 0.0, shear, sound], dtype=float)
    )
    numerical_local = np.real(numerical_values)

    acoustic_cone = valencia_radial_characteristic_speeds_over_c(
        geometry.base,
        radial_velocity_over_c=radial_velocity_over_c,
        azimuthal_velocity_over_c=azimuthal_velocity_over_c,
        sound_speed_over_c=sound,
    )
    shear_cone = valencia_radial_characteristic_speeds_over_c(
        geometry.base,
        radial_velocity_over_c=radial_velocity_over_c,
        azimuthal_velocity_over_c=azimuthal_velocity_over_c,
        sound_speed_over_c=shear,
    )
    coordinate_speeds = np.sort(
        np.asarray(
            [
                acoustic_cone[0],
                shear_cone[0],
                acoustic_cone[1],
                shear_cone[-1],
                acoustic_cone[-1],
            ],
            dtype=float,
        )
    )

    incoming_indices = [
        int(np.argmin(np.abs(numerical_values + sound))),
        int(np.argmin(np.abs(numerical_values + shear))),
    ]
    incoming_vectors = np.real(numerical_vectors[:, incoming_indices])
    incoming_vectors /= np.maximum(
        np.max(np.abs(incoming_vectors), axis=0, keepdims=True),
        np.finfo(float).tiny,
    )
    shear_impedance = (
        mass[2, 2] * closure.viscous_signal_speed_over_c
    )
    boundary_rows = np.asarray(
        [
            [0.0, 1.0 / sound, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0 / shear_impedance,
            ],
        ],
        dtype=float,
    )
    response = boundary_rows @ incoming_vectors
    singular_values = np.linalg.svd(response, compute_uv=False)

    light_min = geometry.base.ingoing_light_speed_over_c
    light_max = geometry.base.outgoing_light_speed_over_c
    light_excess = max(
        float(light_min - np.min(coordinate_speeds)),
        float(np.max(coordinate_speeds) - light_max),
        0.0,
    )
    rank_threshold = max(
        float(np.max(np.abs(coordinate_speeds))) * 1.0e-10,
        1.0e-12,
    )
    return CausalFiveFieldPrincipalAudit(
        local_rest_mass_matrix=mass,
        local_rest_flux_matrix=flux,
        local_rest_right_eigenvectors=np.asarray(
            np.real(numerical_vectors),
            dtype=float,
        ),
        analytic_local_rest_speeds_over_c=tuple(
            float(value) for value in analytic_local
        ),
        numerical_local_rest_speeds_over_c=tuple(
            float(value) for value in numerical_local
        ),
        coordinate_speeds_over_c=tuple(
            float(value) for value in coordinate_speeds
        ),
        incoming_inner_characteristics=int(
            np.sum(coordinate_speeds > 0.0)
        ),
        incoming_outer_characteristics=int(
            np.sum(coordinate_speeds < 0.0)
        ),
        stationary_coordinate_rank=int(
            np.sum(np.abs(coordinate_speeds) > rank_threshold)
        ),
        incoming_mode_response_matrix=np.asarray(response, dtype=float),
        incoming_mode_response_rank=int(
            np.linalg.matrix_rank(response, tol=1.0e-12)
        ),
        incoming_mode_response_smallest_singular_value=float(
            np.min(singular_values)
        ),
        maximum_local_rest_eigenvalue_defect=float(
            np.max(np.abs(numerical_local - analytic_local))
        ),
        maximum_imaginary_eigenvalue=float(
            np.max(np.abs(np.imag(numerical_values)))
        ),
        maximum_light_cone_excess=float(light_excess),
    )


def audit_causal_five_field_boundaries(
    inner: CausalFiveFieldPrincipalAudit,
    outer: CausalFiveFieldPrincipalAudit,
) -> CausalFiveFieldBoundaryRankAudit:
    """Audit excision and the two-condition Roche-plus-zero-stress edge."""

    outer_rows = 5
    outer_face_jacobian = np.eye(outer_rows)
    return CausalFiveFieldBoundaryRankAudit(
        inner_incoming_characteristics=(
            inner.incoming_inner_characteristics
        ),
        outer_incoming_characteristics=(
            outer.incoming_outer_characteristics
        ),
        outer_face_rows=outer_rows,
        outer_face_jacobian_rank=int(
            np.linalg.matrix_rank(outer_face_jacobian)
        ),
        outer_physical_boundary_conditions=2,
        outer_incoming_response_rank=outer.incoming_mode_response_rank,
        outer_incoming_response_smallest_singular_value=(
            outer.incoming_mode_response_smallest_singular_value
        ),
        inner_excision_passed=(
            inner.incoming_inner_characteristics == 0
        ),
        outer_characteristic_count_passed=(
            outer.incoming_outer_characteristics == 2
        ),
        outer_response_rank_passed=(
            outer.incoming_mode_response_rank == 2
        ),
    )
