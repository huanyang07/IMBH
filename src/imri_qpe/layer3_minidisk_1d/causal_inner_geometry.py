"""Geometric finite-volume terms for the causal Kerr-Schild column."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C

from .causal_inner_valencia import (
    SchwarzschildKerrSchildGeometry,
    ValenciaColumnState,
    valencia_column_state,
)


@dataclass(frozen=True)
class KerrSchildColumnGeometry:
    """Equatorial 2+1 Kerr-Schild column geometry and radial derivatives."""

    base: SchwarzschildKerrSchildGeometry
    spacetime_metric: np.ndarray
    inverse_spacetime_metric: np.ndarray
    radial_spacetime_metric_derivative: np.ndarray
    lapse_derivative: float
    radial_shift_derivative_over_c: float
    extrinsic_curvature: np.ndarray

    @property
    def radius(self) -> float:
        return self.base.radius

    @property
    def gravitational_radius(self) -> float:
        return self.base.gravitational_radius

    @property
    def face_measure(self) -> float:
        return self.base.proper_column_jacobian


@dataclass(frozen=True)
class ValenciaPerfectFluidPrimitive:
    """One generic perfect-fluid column primitive state."""

    surface_density: float
    radial_velocity_over_c: float
    azimuthal_velocity_over_c: float
    specific_internal_energy: float
    integrated_pressure: float


@dataclass(frozen=True)
class KerrSchildColumnSourceAudit:
    """Local Valencia/Killing transform and geometric-source audit."""

    valencia_state: ValenciaColumnState
    killing_conserved: np.ndarray
    killing_flux_over_c: np.ndarray
    radial_momentum_source: float
    tau_source: float
    momentum_source_identity_defect: float
    tau_source_identity_defect: float
    killing_density_identity_defect: float
    killing_flux_identity_defect: float


@dataclass(frozen=True)
class KerrSchildColumnGrid:
    """Exact radial finite-volume measures for the equatorial column."""

    edges: np.ndarray
    centers: np.ndarray
    cell_measures: np.ndarray
    face_measures: np.ndarray
    gravitational_radius: float


@dataclass(frozen=True)
class KerrSchildStationaryFiniteVolumeAudit:
    """Stationary flux/source residual for one prescribed exact profile."""

    grid: KerrSchildColumnGrid
    weighted_face_fluxes_over_c: np.ndarray
    integrated_geometric_sources: np.ndarray
    integrated_residuals: np.ndarray
    telescoping_defect: np.ndarray


def kerr_schild_column_geometry(
    radius: float,
    gravitational_radius: float,
) -> KerrSchildColumnGeometry:
    """Return the equatorial 2+1 ingoing-Kerr-Schild column geometry."""

    radius = float(radius)
    gravitational_radius = float(gravitational_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    if (
        not np.isfinite(gravitational_radius)
        or gravitational_radius < 0.0
    ):
        raise ValueError("gravitational radius must be finite and non-negative")

    metric_ratio = 2.0 * gravitational_radius / radius
    metric_ratio_derivative = -metric_ratio / radius
    gamma_rr = 1.0 + metric_ratio
    lapse = 1.0 / np.sqrt(gamma_rr)
    radial_shift = metric_ratio / gamma_rr
    lapse_derivative = (
        lapse * metric_ratio / (2.0 * radius * gamma_rr)
    )
    radial_shift_derivative = (
        -metric_ratio / (radius * gamma_rr**2)
    )
    base = SchwarzschildKerrSchildGeometry(
        radius=radius,
        gravitational_radius=gravitational_radius,
        lapse=float(lapse),
        radial_shift_over_c=float(radial_shift),
        gamma_rr=float(gamma_rr),
        gamma_phiphi=float(radius**2),
        inverse_gamma_rr=float(1.0 / gamma_rr),
        sqrt_spatial_metric=float(np.sqrt(gamma_rr) * radius**2),
    )

    spacetime_metric = np.asarray(
        [
            [metric_ratio - 1.0, metric_ratio, 0.0],
            [metric_ratio, gamma_rr, 0.0],
            [0.0, 0.0, radius**2],
        ],
        dtype=float,
    )
    inverse_spacetime_metric = np.asarray(
        [
            [-gamma_rr, metric_ratio, 0.0],
            [metric_ratio, 1.0 - metric_ratio, 0.0],
            [0.0, 0.0, 1.0 / radius**2],
        ],
        dtype=float,
    )
    radial_metric_derivative = np.asarray(
        [
            [
                metric_ratio_derivative,
                metric_ratio_derivative,
                0.0,
            ],
            [
                metric_ratio_derivative,
                metric_ratio_derivative,
                0.0,
            ],
            [0.0, 0.0, 2.0 * radius],
        ],
        dtype=float,
    )

    radial_christoffel_rr = (
        0.5 * base.inverse_gamma_rr * metric_ratio_derivative
    )
    covariant_radial_shift = metric_ratio
    covariant_shift_derivative = metric_ratio_derivative
    extrinsic_rr = (
        covariant_shift_derivative
        - radial_christoffel_rr * covariant_radial_shift
    ) / lapse
    extrinsic_phiphi = (
        radius * covariant_radial_shift / (lapse * gamma_rr)
    )
    extrinsic_curvature = np.asarray(
        [
            [extrinsic_rr, 0.0],
            [0.0, extrinsic_phiphi],
        ],
        dtype=float,
    )
    return KerrSchildColumnGeometry(
        base=base,
        spacetime_metric=spacetime_metric,
        inverse_spacetime_metric=inverse_spacetime_metric,
        radial_spacetime_metric_derivative=radial_metric_derivative,
        lapse_derivative=float(lapse_derivative),
        radial_shift_derivative_over_c=float(radial_shift_derivative),
        extrinsic_curvature=extrinsic_curvature,
    )


def kerr_schild_column_measure_antiderivative(
    radius: float,
    gravitational_radius: float,
) -> float:
    """Return an antiderivative of ``2 pi R sqrt(gamma_RR)``."""

    radius = float(radius)
    gravitational_radius = float(gravitational_radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if gravitational_radius < 0.0:
        raise ValueError("gravitational radius cannot be negative")
    if gravitational_radius == 0.0:
        return float(np.pi * radius**2)
    root = np.sqrt(radius * (radius + 2.0 * gravitational_radius))
    argument = (radius + gravitational_radius) / gravitational_radius
    primitive = (
        (radius + gravitational_radius) * root
        - gravitational_radius**2 * np.arccosh(argument)
    )
    return float(np.pi * primitive)


def make_kerr_schild_column_grid(
    inner_radius: float,
    outer_radius: float,
    n_cells: int,
    gravitational_radius: float,
) -> KerrSchildColumnGrid:
    """Return a logarithmic grid with exact proper column measures."""

    if inner_radius <= 0.0 or outer_radius <= inner_radius:
        raise ValueError("column grid radii are invalid")
    if int(n_cells) != n_cells or n_cells < 1:
        raise ValueError("column grid requires a positive integer cell count")
    if gravitational_radius < 0.0:
        raise ValueError("gravitational radius cannot be negative")
    edges = np.geomspace(
        float(inner_radius),
        float(outer_radius),
        int(n_cells) + 1,
    )
    return make_kerr_schild_column_grid_from_edges(
        edges,
        gravitational_radius,
    )


def make_kerr_schild_column_grid_from_edges(
    edges: np.ndarray,
    gravitational_radius: float,
) -> KerrSchildColumnGrid:
    """Return an exact column grid for arbitrary increasing radial edges.

    Cell centers remain logarithmic midpoints, matching the primitive
    reconstruction coordinate used by the causal DAE.  The cell and face
    measures are evaluated from the exact Kerr--Schild geometry, so a
    nonuniform fine/coarse grid retains exact finite-volume telescoping.
    """

    edges = np.asarray(edges, dtype=float)
    gravitational_radius = float(gravitational_radius)
    if (
        edges.ndim != 1
        or edges.size < 2
        or np.any(~np.isfinite(edges))
        or edges[0] <= 0.0
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("column grid edges must be finite and increasing")
    if (
        not np.isfinite(gravitational_radius)
        or gravitational_radius < 0.0
    ):
        raise ValueError("gravitational radius cannot be negative")
    centers = np.sqrt(edges[:-1] * edges[1:])
    primitive = np.asarray(
        [
            kerr_schild_column_measure_antiderivative(
                radius,
                gravitational_radius,
            )
            for radius in edges
        ],
        dtype=float,
    )
    cell_measures = np.diff(primitive)
    face_measures = np.asarray(
        [
            kerr_schild_column_geometry(
                radius,
                gravitational_radius,
            ).face_measure
            for radius in edges
        ],
        dtype=float,
    )
    if np.any(cell_measures <= 0.0):
        raise ValueError("column cell measures must be positive")
    return KerrSchildColumnGrid(
        edges=edges,
        centers=centers,
        cell_measures=cell_measures,
        face_measures=face_measures,
        gravitational_radius=float(gravitational_radius),
    )


def _column_stress_energy(
    geometry: KerrSchildColumnGeometry,
    primitive: ValenciaPerfectFluidPrimitive,
    state: ValenciaColumnState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return four-velocity, contravariant stress, and spatial stress."""

    sigma = float(primitive.surface_density)
    pressure_mass = float(primitive.integrated_pressure) / C**2
    lorentz = state.lorentz_factor
    enthalpy = state.specific_enthalpy_over_c2
    coordinate_v_r = (
        primitive.radial_velocity_over_c
        / np.sqrt(geometry.base.gamma_rr)
    )
    coordinate_v_phi = (
        primitive.azimuthal_velocity_over_c / geometry.radius
    )
    four_velocity = np.asarray(
        [
            lorentz / geometry.base.lapse,
            lorentz
            * (
                coordinate_v_r
                - geometry.base.radial_shift_over_c
                / geometry.base.lapse
            ),
            lorentz * coordinate_v_phi,
        ],
        dtype=float,
    )
    stress_energy = (
        sigma * enthalpy * np.outer(four_velocity, four_velocity)
        + pressure_mass * geometry.inverse_spacetime_metric
    )
    spatial_inverse_metric = np.asarray(
        [
            [geometry.base.inverse_gamma_rr, 0.0],
            [0.0, 1.0 / geometry.base.gamma_phiphi],
        ],
        dtype=float,
    )
    spatial_velocity = np.asarray(
        [coordinate_v_r, coordinate_v_phi],
        dtype=float,
    )
    spatial_stress = (
        sigma
        * enthalpy
        * lorentz**2
        * np.outer(spatial_velocity, spatial_velocity)
        + pressure_mass * spatial_inverse_metric
    )
    return four_velocity, stress_energy, spatial_stress


def audit_kerr_schild_column_sources(
    geometry: KerrSchildColumnGeometry,
    primitive: ValenciaPerfectFluidPrimitive,
) -> KerrSchildColumnSourceAudit:
    """Return local geometric sources and independent identity defects."""

    state = valencia_column_state(
        geometry.base,
        surface_density=primitive.surface_density,
        radial_velocity_over_c=primitive.radial_velocity_over_c,
        azimuthal_velocity_over_c=primitive.azimuthal_velocity_over_c,
        specific_internal_energy=primitive.specific_internal_energy,
        integrated_pressure=primitive.integrated_pressure,
    )
    four_velocity, stress_energy, spatial_stress = _column_stress_energy(
        geometry,
        primitive,
        state,
    )
    del four_velocity

    alpha = geometry.base.lapse
    shift = geometry.base.radial_shift_over_c
    energy_density = state.conserved[3] + state.conserved[0]
    killing_density = (
        alpha * energy_density - shift * state.conserved[1]
    )
    killing_flux = (
        alpha * (state.flux_over_c[3] + state.flux_over_c[0])
        - shift * state.flux_over_c[1]
    )
    killing_conserved = np.asarray(
        [
            state.conserved[0],
            state.conserved[1],
            state.conserved[2],
            killing_density,
        ],
        dtype=float,
    )
    killing_fluxes = np.asarray(
        [
            state.flux_over_c[0],
            state.flux_over_c[1],
            state.flux_over_c[2],
            killing_flux,
        ],
        dtype=float,
    )

    direct_radial_source = 0.5 * alpha * float(
        np.sum(
            stress_energy
            * geometry.radial_spacetime_metric_derivative
        )
    )
    contravariant_radial_momentum = (
        geometry.base.inverse_gamma_rr * state.conserved[1]
    )
    momentum_source_3p1 = (
        -energy_density * geometry.lapse_derivative
        + state.conserved[1]
        * geometry.radial_shift_derivative_over_c
        + 0.5
        * alpha
        * (
            spatial_stress[0, 0]
            * geometry.radial_spacetime_metric_derivative[1, 1]
            + spatial_stress[1, 1]
            * geometry.radial_spacetime_metric_derivative[2, 2]
        )
    )

    direct_tau_source = alpha * (
        float(np.sum(spatial_stress * geometry.extrinsic_curvature))
        - contravariant_radial_momentum
        * geometry.lapse_derivative
        / alpha
    )
    killing_tau_source = (
        shift * direct_radial_source
        - geometry.lapse_derivative
        * (state.flux_over_c[3] + state.flux_over_c[0])
        + geometry.radial_shift_derivative_over_c
        * state.flux_over_c[1]
    ) / alpha

    direct_killing_density = -alpha * float(
        np.dot(stress_energy[0], geometry.spacetime_metric[:, 0])
    )
    direct_killing_flux = -alpha * float(
        np.dot(stress_energy[1], geometry.spacetime_metric[:, 0])
    )
    return KerrSchildColumnSourceAudit(
        valencia_state=state,
        killing_conserved=killing_conserved,
        killing_flux_over_c=killing_fluxes,
        radial_momentum_source=float(direct_radial_source),
        tau_source=float(killing_tau_source),
        momentum_source_identity_defect=float(
            direct_radial_source - momentum_source_3p1
        ),
        tau_source_identity_defect=float(
            killing_tau_source - direct_tau_source
        ),
        killing_density_identity_defect=float(
            killing_density - direct_killing_density
        ),
        killing_flux_identity_defect=float(
            killing_flux - direct_killing_flux
        ),
    )


def valencia_conserved_from_killing(
    geometry: KerrSchildColumnGeometry,
    killing_conserved,
) -> np.ndarray:
    """Transform ``(D,S_R,S_phi,E_K)`` back to local Valencia variables."""

    conserved = np.asarray(killing_conserved, dtype=float)
    if conserved.shape != (4,) or np.any(~np.isfinite(conserved)):
        raise ValueError("Killing conserved state must be finite and length four")
    rest_mass, radial_momentum, angular_momentum, killing_energy = conserved
    tau = (
        killing_energy
        + geometry.base.radial_shift_over_c * radial_momentum
    ) / geometry.base.lapse - rest_mass
    return np.asarray(
        [rest_mass, radial_momentum, angular_momentum, tau],
        dtype=float,
    )


def _integrated_radial_source(
    left_radius: float,
    right_radius: float,
    gravitational_radius: float,
    primitive_provider: Callable[[float], ValenciaPerfectFluidPrimitive],
    quadrature_order: int,
) -> float:
    if int(quadrature_order) != quadrature_order or quadrature_order < 1:
        raise ValueError("quadrature order must be a positive integer")
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    midpoint = 0.5 * (left_radius + right_radius)
    half_width = 0.5 * (right_radius - left_radius)
    integral = 0.0
    for node, weight in zip(nodes, weights, strict=True):
        radius = midpoint + half_width * float(node)
        geometry = kerr_schild_column_geometry(
            radius,
            gravitational_radius,
        )
        source = audit_kerr_schild_column_sources(
            geometry,
            primitive_provider(radius),
        ).radial_momentum_source
        integral += float(weight) * geometry.face_measure * source
    return float(half_width * integral)


def audit_stationary_kerr_schild_finite_volume_profile(
    grid: KerrSchildColumnGrid,
    primitive_provider: Callable[[float], ValenciaPerfectFluidPrimitive],
    *,
    quadrature_order: int = 8,
) -> KerrSchildStationaryFiniteVolumeAudit:
    """Audit a prescribed stationary profile in the Killing-energy chart."""

    weighted_face_fluxes = np.empty(
        (grid.edges.size, 4),
        dtype=float,
    )
    for index, radius in enumerate(grid.edges):
        geometry = kerr_schild_column_geometry(
            float(radius),
            grid.gravitational_radius,
        )
        audit = audit_kerr_schild_column_sources(
            geometry,
            primitive_provider(float(radius)),
        )
        weighted_face_fluxes[index] = (
            geometry.face_measure * audit.killing_flux_over_c
        )

    integrated_sources = np.zeros(
        (grid.centers.size, 4),
        dtype=float,
    )
    for index, (left_radius, right_radius) in enumerate(
        zip(grid.edges[:-1], grid.edges[1:], strict=True)
    ):
        integrated_sources[index, 1] = _integrated_radial_source(
            float(left_radius),
            float(right_radius),
            grid.gravitational_radius,
            primitive_provider,
            quadrature_order,
        )
    flux_differences = np.diff(weighted_face_fluxes, axis=0)
    residuals = flux_differences - integrated_sources
    telescoping_defect = (
        np.sum(flux_differences, axis=0)
        - (weighted_face_fluxes[-1] - weighted_face_fluxes[0])
    )
    return KerrSchildStationaryFiniteVolumeAudit(
        grid=grid,
        weighted_face_fluxes_over_c=weighted_face_fluxes,
        integrated_geometric_sources=integrated_sources,
        integrated_residuals=residuals,
        telescoping_defect=telescoping_defect,
    )
