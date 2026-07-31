"""High-order continuum reference for the causal five-field radial DAE.

This module is diagnostic only.  It does not change the finite-volume
operator.  It evaluates the same declared local physical maps on a smooth
log-radius collocation grid and differentiates their spline interpolants.
The result supplies an independent continuum action against which the
cell-integrated monolithic tangent can be compared block by block.

The linearized temporal action includes both pieces required away from a
stationary base state,

    T(p) delta_p_t + T_,p[delta_p] p_t.

Omitting the second term would audit a different DAE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
from scipy.sparse import block_diag, csr_matrix, eye, kron

from imri_qpe.constants import C

from .causal_inner_dae_system import CausalFiveFieldDAEContext
from .causal_inner_geometry import kerr_schild_column_geometry
from .causal_inner_monolithic_tangent import (
    CausalFiveFieldMonolithicFrozenTangent,
)
from .causal_inner_radial_linear_tangent import (
    causal_five_field_analytic_local_maps,
)


_N_FIELDS = 5

CONTINUUM_DAE_BLOCK_NAMES = (
    "mapped_temporal",
    "responsive_height_temporal",
    "mapped_storage_rate",
    "responsive_height_storage_rate",
    "candidate_conservative_transport",
    "candidate_shear_principal",
    "candidate_height_principal",
    "candidate_local_stress_relaxation",
    "candidate_geometry",
    "candidate_cooling",
    "candidate_stream",
    "candidate_lower_height_work",
)


def _spline(
    log_radii: np.ndarray,
    values: np.ndarray,
) -> BSpline:
    return make_interp_spline(
        np.asarray(log_radii, dtype=float),
        np.asarray(values, dtype=float),
        k=5,
        axis=0,
    )


def _log_derivative(
    log_radii: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        _spline(log_radii, values).derivative()(log_radii),
        dtype=float,
    )


@dataclass(frozen=True)
class CausalFiveFieldContinuumBackground:
    """Smooth collocation representation of one continuum base state."""

    context: CausalFiveFieldDAEContext
    log_radii: np.ndarray
    radii: np.ndarray
    face_measures: np.ndarray
    primitive_charts: np.ndarray
    primitive_radius_derivative: np.ndarray
    mapped_conserved_jacobians: np.ndarray
    mapped_conserved_hessians: np.ndarray
    vertical_storage_matrices: np.ndarray
    vertical_storage_derivatives: np.ndarray
    temporal_storage_matrices: np.ndarray
    physical_flux_jacobians: np.ndarray
    shear_principal_matrices: np.ndarray
    shear_principal_derivatives: np.ndarray
    height_principal_matrices: np.ndarray
    height_principal_derivatives: np.ndarray
    lower_source_values: dict[str, np.ndarray]
    lower_source_jacobians: dict[str, np.ndarray]
    base_stationary_block_densities: dict[str, np.ndarray]
    base_stationary_density: np.ndarray
    base_rate_per_s: np.ndarray


@dataclass(frozen=True)
class CausalFiveFieldContinuumLinearizedReference:
    """One smooth continuum perturbation and its exact DAE block action."""

    background: CausalFiveFieldContinuumBackground
    perturbation: np.ndarray
    perturbation_radius_derivative: np.ndarray
    perturbation_rate_per_s: np.ndarray
    face_flux_jvp: np.ndarray
    block_densities_per_radius: dict[str, np.ndarray]
    total_density_per_radius: np.ndarray
    maximum_pointwise_ledger_relative_defect: float

    def evaluate_rate(self, radii: np.ndarray) -> np.ndarray:
        """Evaluate the collocation perturbation rate at positive radii."""

        values = np.asarray(radii, dtype=float)
        if values.ndim != 1 or np.any(values <= 0.0):
            raise ValueError("continuum rate radii must be positive")
        result = np.asarray(
            _spline(
                self.background.log_radii,
                self.perturbation_rate_per_s,
            )(np.log(values)),
            dtype=float,
        )
        if result.shape != (values.size, _N_FIELDS):
            raise RuntimeError("continuum rate interpolation failed")
        return result

    def evaluate_face_flux_jvp(self, radii: np.ndarray) -> np.ndarray:
        """Evaluate the physical shared-flux JVP at selected faces."""

        values = np.asarray(radii, dtype=float)
        if values.ndim != 1 or np.any(values <= 0.0):
            raise ValueError("continuum face radii must be positive")
        result = np.asarray(
            _spline(
                self.background.log_radii,
                self.face_flux_jvp,
            )(np.log(values)),
            dtype=float,
        )
        if result.shape != (values.size, _N_FIELDS):
            raise RuntimeError("continuum face interpolation failed")
        return result

    def integrate_blocks(self, edges: np.ndarray) -> dict[str, np.ndarray]:
        """Integrate every continuum DAE block over declared radial cells."""

        radial_edges = np.asarray(edges, dtype=float)
        if (
            radial_edges.ndim != 1
            or radial_edges.size < 2
            or np.any(radial_edges <= 0.0)
            or np.any(np.diff(radial_edges) <= 0.0)
        ):
            raise ValueError("continuum integration edges are invalid")
        lower = float(self.background.radii[0])
        upper = float(self.background.radii[-1])
        tolerance = 1.0e-12 * max(abs(lower), abs(upper), 1.0)
        if (
            radial_edges[0] < lower - tolerance
            or radial_edges[-1] > upper + tolerance
        ):
            raise ValueError("continuum integration leaves reference domain")
        log_edges = np.log(radial_edges)
        result: dict[str, np.ndarray] = {}
        radii = self.background.radii
        for name in CONTINUUM_DAE_BLOCK_NAMES:
            # The stored density is per unit radius.  Multiplication by R
            # converts it to a density per unit log radius.
            interpolant = _spline(
                self.background.log_radii,
                self.block_densities_per_radius[name] * radii[:, None],
            )
            rows = [
                np.asarray(
                    interpolant.integrate(left, right),
                    dtype=float,
                )
                for left, right in zip(
                    log_edges[:-1],
                    log_edges[1:],
                    strict=True,
                )
            ]
            result[name] = np.asarray(rows, dtype=float)
        return result


@dataclass(frozen=True)
class CausalFiveFieldDiscreteTruncation:
    """Cellwise discrete-minus-continuum DAE truncation ledger."""

    block_rows: dict[str, np.ndarray]
    total_rows: np.ndarray
    mass_solved_scaled_rate_error: np.ndarray
    discrete_block_rows: dict[str, np.ndarray]
    continuum_block_rows: dict[str, np.ndarray]
    discrete_face_flux_jvp: np.ndarray
    continuum_face_flux_jvp: np.ndarray
    maximum_discrete_ledger_relative_defect: float
    maximum_continuum_ledger_relative_defect: float
    maximum_truncation_ledger_relative_defect: float


def build_causal_five_field_continuum_background(
    context: CausalFiveFieldDAEContext,
    background_evaluator: Callable[[np.ndarray], np.ndarray],
    *,
    node_count: int,
) -> CausalFiveFieldContinuumBackground:
    """Build one high-order continuum base-state collocation reference."""

    context = context.validated()
    count = int(node_count)
    if count < 17:
        raise ValueError("continuum reference needs at least 17 nodes")
    lower = float(np.log(context.grid.edges[0]))
    upper = float(np.log(context.grid.edges[-1]))
    log_radii = np.linspace(lower, upper, count, dtype=float)
    radii = np.exp(log_radii)
    charts = np.asarray(background_evaluator(radii), dtype=float)
    if (
        charts.shape != (count, _N_FIELDS)
        or np.any(~np.isfinite(charts))
    ):
        raise ValueError("continuum background evaluation failed")
    chart_radius_derivative = (
        _log_derivative(log_radii, charts) / radii[:, None]
    )

    local_maps = [
        causal_five_field_analytic_local_maps(
            context,
            float(radius),
            chart,
        )
        for radius, chart in zip(radii, charts, strict=True)
    ]
    face_measures = np.asarray(
        [
            kerr_schild_column_geometry(
                float(radius),
                context.grid.gravitational_radius,
            ).face_measure
            for radius in radii
        ],
        dtype=float,
    )
    mapped_jacobians = np.asarray(
        [local.mapped_conserved_jacobian for local in local_maps],
        dtype=float,
    )
    mapped_hessians = np.asarray(
        [local.mapped_conserved_hessian for local in local_maps],
        dtype=float,
    )
    height_matrices = np.asarray(
        [local.vertical_storage_matrix for local in local_maps],
        dtype=float,
    )
    height_derivatives = np.asarray(
        [local.vertical_storage_derivative for local in local_maps],
        dtype=float,
    )
    temporal = np.asarray(
        [local.temporal_storage_matrix for local in local_maps],
        dtype=float,
    )
    fluxes = np.asarray(
        [local.physical_flux_over_c for local in local_maps],
        dtype=float,
    )
    flux_jacobians = np.asarray(
        [local.physical_flux_jacobian for local in local_maps],
        dtype=float,
    )
    shear = np.asarray(
        [local.shear_principal_source_matrix for local in local_maps],
        dtype=float,
    )
    shear_derivative = np.asarray(
        [
            local.shear_principal_source_derivative
            for local in local_maps
        ],
        dtype=float,
    )
    height = np.asarray(
        [local.vertical_principal_source_matrix for local in local_maps],
        dtype=float,
    )
    height_derivative = np.asarray(
        [
            local.vertical_principal_source_derivative
            for local in local_maps
        ],
        dtype=float,
    )
    lower_names = tuple(local_maps[0].lower_source_values)
    lower_values = {
        name: np.asarray(
            [local.lower_source_values[name] for local in local_maps],
            dtype=float,
        )
        for name in lower_names
    }
    lower_jacobians = {
        name: np.asarray(
            [local.lower_source_jacobians[name] for local in local_maps],
            dtype=float,
        )
        for name in lower_names
    }

    conservative = (
        _log_derivative(
            log_radii,
            face_measures[:, None] * fluxes,
        )
        / radii[:, None]
    )
    base_blocks = {
        "candidate_conservative_transport": conservative,
        "candidate_shear_principal": (
            -face_measures[:, None]
            * np.einsum(
                "nij,nj->ni",
                shear,
                chart_radius_derivative,
            )
        ),
        "candidate_height_principal": (
            -face_measures[:, None]
            * np.einsum(
                "nij,nj->ni",
                height,
                chart_radius_derivative,
            )
        ),
        "candidate_local_stress_relaxation": (
            -face_measures[:, None]
            * lower_values["stress_relaxation"]
        ),
        "candidate_geometry": (
            -face_measures[:, None]
            * (
                lower_values["perfect_fluid_geometry"]
                + lower_values["stress_geometry"]
            )
        ),
        "candidate_cooling": (
            -face_measures[:, None]
            * lower_values["radiative_cooling"]
        ),
        "candidate_stream": np.zeros_like(charts),
        "candidate_lower_height_work": (
            -face_measures[:, None]
            * lower_values["vertical_work"]
        ),
    }
    stationary = sum(
        base_blocks.values(),
        start=np.zeros_like(charts),
    )
    base_rate = -C * np.linalg.solve(
        temporal,
        (stationary / face_measures[:, None])[:, :, None],
    )[:, :, 0]
    return CausalFiveFieldContinuumBackground(
        context=context,
        log_radii=log_radii,
        radii=radii,
        face_measures=face_measures,
        primitive_charts=charts,
        primitive_radius_derivative=chart_radius_derivative,
        mapped_conserved_jacobians=mapped_jacobians,
        mapped_conserved_hessians=mapped_hessians,
        vertical_storage_matrices=height_matrices,
        vertical_storage_derivatives=height_derivatives,
        temporal_storage_matrices=temporal,
        physical_flux_jacobians=flux_jacobians,
        shear_principal_matrices=shear,
        shear_principal_derivatives=shear_derivative,
        height_principal_matrices=height,
        height_principal_derivatives=height_derivative,
        lower_source_values=lower_values,
        lower_source_jacobians=lower_jacobians,
        base_stationary_block_densities=base_blocks,
        base_stationary_density=stationary,
        base_rate_per_s=base_rate,
    )


def causal_sixth_order_inward_collocation_derivative(
    log_radii: np.ndarray,
) -> csr_matrix:
    """Return a sixth-order derivative for an inward-only collocation domain.

    The interior uses a sixth-order seven-point upwind-biased stencil with
    offsets ``[-2,-1,0,1,2,3,4]``.  Its Fourier symbol has nonpositive real
    part for the declared negative-speed system.  The first two outflow rows
    use the stable first-order forward closure.  Missing values beyond the
    outer edge are homogeneous inflow data; the frozen packets vanish
    throughout that buffer.
    """

    nodes = np.asarray(log_radii, dtype=float)
    if (
        nodes.ndim != 1
        or nodes.size < 9
        or np.any(~np.isfinite(nodes))
        or np.any(np.diff(nodes) <= 0.0)
    ):
        raise ValueError("collocation log-radius nodes are invalid")
    spacing = np.diff(nodes)
    step = float(np.mean(spacing))
    if np.max(np.abs(spacing - step)) > 1.0e-12 * max(abs(step), 1.0):
        raise ValueError("collocation derivative requires uniform log nodes")
    rows = []
    columns = []
    values = []
    count = nodes.size
    for row in (0, 1):
        rows.extend((row, row))
        columns.extend((row, row + 1))
        values.extend((-1.0 / step, 1.0 / step))
    offsets = np.arange(-2, 5, dtype=float)
    matrix = np.asarray(
        [offsets**degree for degree in range(7)],
        dtype=float,
    )
    target = np.zeros(7, dtype=float)
    target[1] = 1.0
    weights = np.linalg.solve(matrix, target) / step
    for row in range(2, count):
        for offset, weight in zip(
            offsets.astype(int),
            weights,
            strict=True,
        ):
            column = row + offset
            if column < count:
                rows.append(row)
                columns.append(column)
                values.append(weight)
    return csr_matrix(
        (values, (rows, columns)),
        shape=(count, count),
        dtype=float,
    )


def causal_five_field_inward_collocation_generator(
    background: CausalFiveFieldContinuumBackground,
) -> csr_matrix:
    """Assemble an independent high-order continuum-history generator.

    The returned matrix acts on node-major primitive perturbations and uses
    the complete linearized continuum DAE, including mapped/height storage
    derivatives and all lower-order blocks.  It is intentionally independent
    of the finite-volume face/reconstruction tangent.
    """

    radii = np.asarray(background.radii, dtype=float)
    count = radii.size
    if count < 9:
        raise ValueError("continuum generator needs at least nine nodes")
    spatial = (
        background.physical_flux_jacobians
        - background.shear_principal_matrices
        - background.height_principal_matrices
    )
    maximum_speed = max(
        float(
            np.max(
                np.real(
                    np.linalg.eigvals(
                        np.linalg.solve(
                            background.temporal_storage_matrices[index],
                            spatial[index],
                        )
                    )
                )
            )
        )
        for index in range(count)
    )
    if maximum_speed >= 0.0:
        raise ValueError(
            "inward collocation generator requires strictly negative speeds"
        )

    derivative = causal_sixth_order_inward_collocation_derivative(
        background.log_radii
    )
    field_identity = eye(_N_FIELDS, format="csr")
    derivative_fields = kron(derivative, field_identity, format="csr")

    def local(blocks: np.ndarray) -> csr_matrix:
        values = np.asarray(blocks, dtype=float)
        if values.shape != (count, _N_FIELDS, _N_FIELDS):
            raise ValueError("local continuum blocks have invalid shape")
        return block_diag(tuple(values), format="csr")

    measures = np.asarray(background.face_measures, dtype=float)
    inverse_radius = local(
        np.repeat(
            (1.0 / radii)[:, None, None],
            _N_FIELDS,
            axis=1,
        )
        * np.eye(_N_FIELDS)[None]
    )
    flux_action = local(
        measures[:, None, None]
        * background.physical_flux_jacobians
    )
    conservative = inverse_radius @ derivative_fields @ flux_action

    shear_state = np.einsum(
        "nijk,nj->nik",
        background.shear_principal_derivatives,
        background.primitive_radius_derivative,
        optimize=True,
    )
    height_state = np.einsum(
        "nijk,nj->nik",
        background.height_principal_derivatives,
        background.primitive_radius_derivative,
        optimize=True,
    )
    shear = local(-measures[:, None, None] * shear_state) + local(
        -measures[:, None, None]
        * background.shear_principal_matrices
        / radii[:, None, None]
    ) @ derivative_fields
    height = local(-measures[:, None, None] * height_state) + local(
        -measures[:, None, None]
        * background.height_principal_matrices
        / radii[:, None, None]
    ) @ derivative_fields

    lower = (
        local(
            -measures[:, None, None]
            * background.lower_source_jacobians["stress_relaxation"]
        )
        + local(
            -measures[:, None, None]
            * (
                background.lower_source_jacobians[
                    "perfect_fluid_geometry"
                ]
                + background.lower_source_jacobians["stress_geometry"]
            )
        )
        + local(
            -measures[:, None, None]
            * background.lower_source_jacobians["radiative_cooling"]
        )
        + local(
            -measures[:, None, None]
            * background.lower_source_jacobians["vertical_work"]
        )
    )
    mapped_storage = np.einsum(
        "nijk,nj->nik",
        background.mapped_conserved_hessians,
        background.base_rate_per_s,
        optimize=True,
    )
    height_storage = np.einsum(
        "nijk,nj->nik",
        background.vertical_storage_derivatives,
        background.base_rate_per_s,
        optimize=True,
    )
    storage = local(
        measures[:, None, None] / C * (mapped_storage + height_storage)
    )
    density_action = conservative + shear + height + lower + storage
    temporal_inverse = local(
        np.asarray(
            [
                -C / measures[index]
                * np.linalg.inv(background.temporal_storage_matrices[index])
                for index in range(count)
            ],
            dtype=float,
        )
    )
    generator = (temporal_inverse @ density_action).tocsr()
    if generator.shape != (
        count * _N_FIELDS,
        count * _N_FIELDS,
    ) or np.any(~np.isfinite(generator.data)):
        raise RuntimeError("continuum collocation generator assembly failed")
    return generator


def linearize_causal_five_field_continuum_reference(
    background: CausalFiveFieldContinuumBackground,
    perturbation_evaluator: Callable[[np.ndarray], np.ndarray],
) -> CausalFiveFieldContinuumLinearizedReference:
    """Linearize the smooth continuum DAE around one collocation base."""

    radii = background.radii
    log_radii = background.log_radii
    measures = background.face_measures
    perturbation = np.asarray(
        perturbation_evaluator(radii),
        dtype=float,
    )
    if (
        perturbation.shape != (radii.size, _N_FIELDS)
        or np.any(~np.isfinite(perturbation))
    ):
        raise ValueError("continuum perturbation evaluation failed")
    perturbation_radius_derivative = (
        _log_derivative(log_radii, perturbation) / radii[:, None]
    )
    flux_jvp = measures[:, None] * np.einsum(
        "nij,nj->ni",
        background.physical_flux_jacobians,
        perturbation,
    )
    conservative = (
        _log_derivative(log_radii, flux_jvp) / radii[:, None]
    )

    shear_state_action = np.einsum(
        "nijk,nk->nij",
        background.shear_principal_derivatives,
        perturbation,
    )
    height_state_action = np.einsum(
        "nijk,nk->nij",
        background.height_principal_derivatives,
        perturbation,
    )
    stationary_blocks = {
        "candidate_conservative_transport": conservative,
        "candidate_shear_principal": (
            -measures[:, None]
            * (
                np.einsum(
                    "nij,nj->ni",
                    shear_state_action,
                    background.primitive_radius_derivative,
                )
                + np.einsum(
                    "nij,nj->ni",
                    background.shear_principal_matrices,
                    perturbation_radius_derivative,
                )
            )
        ),
        "candidate_height_principal": (
            -measures[:, None]
            * (
                np.einsum(
                    "nij,nj->ni",
                    height_state_action,
                    background.primitive_radius_derivative,
                )
                + np.einsum(
                    "nij,nj->ni",
                    background.height_principal_matrices,
                    perturbation_radius_derivative,
                )
            )
        ),
        "candidate_local_stress_relaxation": (
            -measures[:, None]
            * np.einsum(
                "nij,nj->ni",
                background.lower_source_jacobians[
                    "stress_relaxation"
                ],
                perturbation,
            )
        ),
        "candidate_geometry": (
            -measures[:, None]
            * np.einsum(
                "nij,nj->ni",
                (
                    background.lower_source_jacobians[
                        "perfect_fluid_geometry"
                    ]
                    + background.lower_source_jacobians[
                        "stress_geometry"
                    ]
                ),
                perturbation,
            )
        ),
        "candidate_cooling": (
            -measures[:, None]
            * np.einsum(
                "nij,nj->ni",
                background.lower_source_jacobians[
                    "radiative_cooling"
                ],
                perturbation,
            )
        ),
        "candidate_stream": np.zeros_like(perturbation),
        "candidate_lower_height_work": (
            -measures[:, None]
            * np.einsum(
                "nij,nj->ni",
                background.lower_source_jacobians["vertical_work"],
                perturbation,
            )
        ),
    }
    mapped_state_action = np.einsum(
        "nijk,nk->nij",
        background.mapped_conserved_hessians,
        perturbation,
    )
    height_state_action = np.einsum(
        "nijk,nk->nij",
        background.vertical_storage_derivatives,
        perturbation,
    )
    mapped_storage_rate = (
        measures[:, None]
        / C
        * np.einsum(
            "nij,nj->ni",
            mapped_state_action,
            background.base_rate_per_s,
        )
    )
    height_storage_rate = (
        measures[:, None]
        / C
        * np.einsum(
            "nij,nj->ni",
            height_state_action,
            background.base_rate_per_s,
        )
    )
    evolving_without_temporal = (
        sum(
            stationary_blocks.values(),
            start=np.zeros_like(perturbation),
        )
        + mapped_storage_rate
        + height_storage_rate
    )
    perturbation_rate = -C * np.linalg.solve(
        background.temporal_storage_matrices,
        (evolving_without_temporal / measures[:, None])[:, :, None],
    )[:, :, 0]
    mapped_temporal = (
        measures[:, None]
        / C
        * np.einsum(
            "nij,nj->ni",
            background.mapped_conserved_jacobians,
            perturbation_rate,
        )
    )
    height_temporal = (
        measures[:, None]
        / C
        * np.einsum(
            "nij,nj->ni",
            background.vertical_storage_matrices,
            perturbation_rate,
        )
    )
    blocks = {
        "mapped_temporal": mapped_temporal,
        "responsive_height_temporal": height_temporal,
        "mapped_storage_rate": mapped_storage_rate,
        "responsive_height_storage_rate": height_storage_rate,
        **stationary_blocks,
    }
    total = sum(
        (blocks[name] for name in CONTINUUM_DAE_BLOCK_NAMES),
        start=np.zeros_like(perturbation),
    )
    block_scale = max(
        *(
            float(np.max(np.abs(block)))
            for block in blocks.values()
        ),
        np.finfo(float).tiny,
    )
    return CausalFiveFieldContinuumLinearizedReference(
        background=background,
        perturbation=perturbation,
        perturbation_radius_derivative=perturbation_radius_derivative,
        perturbation_rate_per_s=perturbation_rate,
        face_flux_jvp=flux_jvp,
        block_densities_per_radius=blocks,
        total_density_per_radius=total,
        maximum_pointwise_ledger_relative_defect=float(
            np.max(np.abs(total)) / block_scale
        ),
    )


def causal_five_field_discrete_dae_truncation(
    tangent: CausalFiveFieldMonolithicFrozenTangent,
    scaled_direction: np.ndarray,
    continuum_rate_cell_averages: np.ndarray,
    continuum_block_rows: dict[str, np.ndarray],
    continuum_face_flux_jvp: np.ndarray,
) -> CausalFiveFieldDiscreteTruncation:
    """Compare one discrete monolithic DAE action with its continuum action."""

    direction = np.asarray(scaled_direction, dtype=float).ravel()
    n_cells = int(tangent.base_primitives.shape[0])
    dimensions = _N_FIELDS * n_cells
    rates = np.asarray(continuum_rate_cell_averages, dtype=float)
    continuum_faces = np.asarray(continuum_face_flux_jvp, dtype=float)
    if (
        direction.shape != (dimensions,)
        or rates.shape != (n_cells, _N_FIELDS)
        or continuum_faces.shape != (n_cells + 1, _N_FIELDS)
        or set(continuum_block_rows) != set(CONTINUUM_DAE_BLOCK_NAMES)
    ):
        raise ValueError("discrete truncation inputs are invalid")
    columns = np.asarray(
        tangent.primitive_column_scales,
        dtype=float,
    )
    rows = np.asarray(
        tangent.conservation_row_scales,
        dtype=float,
    )
    scaled_rate = rates.ravel() / columns

    def physical_rows(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return np.asarray(
            (matrix @ vector) * rows,
            dtype=float,
        ).reshape(n_cells, _N_FIELDS)

    spatial = tangent.spatial_tangent
    discrete_blocks = {
        "mapped_temporal": physical_rows(
            tangent.mapped_descriptor_scaled_matrix,
            scaled_rate,
        ),
        "responsive_height_temporal": physical_rows(
            tangent.responsive_height_descriptor_scaled_matrix,
            scaled_rate,
        ),
        "mapped_storage_rate": physical_rows(
            tangent.mapped_storage_rate_derivative_scaled_matrix,
            direction,
        ),
        "responsive_height_storage_rate": physical_rows(
            (
                tangent
                .responsive_height_storage_rate_derivative_scaled_matrix
            ),
            direction,
        ),
    }
    for name in CONTINUUM_DAE_BLOCK_NAMES[4:]:
        discrete_blocks[name] = physical_rows(
            spatial.block_scaled_jacobians[name],
            direction,
        )

    continuum = {
        name: np.asarray(continuum_block_rows[name], dtype=float)
        for name in CONTINUUM_DAE_BLOCK_NAMES
    }
    if any(
        values.shape != (n_cells, _N_FIELDS)
        for values in continuum.values()
    ):
        raise ValueError("continuum block rows have wrong shape")
    truncation = {
        name: discrete_blocks[name] - continuum[name]
        for name in CONTINUUM_DAE_BLOCK_NAMES
    }
    discrete_total = sum(
        discrete_blocks.values(),
        start=np.zeros((n_cells, _N_FIELDS), dtype=float),
    )
    continuum_total = sum(
        continuum.values(),
        start=np.zeros_like(discrete_total),
    )
    truncation_total = sum(
        truncation.values(),
        start=np.zeros_like(discrete_total),
    )
    direct_discrete = (
        tangent.descriptor_scaled_matrix @ scaled_rate
        + tangent.evolving_scaled_jacobian @ direction
    )
    direct_discrete_physical = (
        direct_discrete * rows
    ).reshape(n_cells, _N_FIELDS)
    scale = max(
        float(np.linalg.norm(discrete_total)),
        float(np.linalg.norm(continuum_total)),
        float(np.linalg.norm(truncation_total)),
        np.finfo(float).tiny,
    )
    truncation_scaled = truncation_total.ravel() / rows
    rate_error = np.linalg.solve(
        tangent.descriptor_scaled_matrix,
        truncation_scaled,
    )
    discrete_faces = np.einsum(
        "fij,j->fi",
        spatial.shared_face_flux_scaled_jacobians,
        direction,
    )
    return CausalFiveFieldDiscreteTruncation(
        block_rows=truncation,
        total_rows=truncation_total,
        mass_solved_scaled_rate_error=rate_error,
        discrete_block_rows=discrete_blocks,
        continuum_block_rows=continuum,
        discrete_face_flux_jvp=np.asarray(discrete_faces, dtype=float),
        continuum_face_flux_jvp=continuum_faces,
        maximum_discrete_ledger_relative_defect=float(
            np.linalg.norm(discrete_total - direct_discrete_physical)
            / scale
        ),
        maximum_continuum_ledger_relative_defect=float(
            np.linalg.norm(continuum_total) / scale
        ),
        maximum_truncation_ledger_relative_defect=float(
            np.linalg.norm(
                truncation_total
                - (discrete_total - continuum_total)
            )
            / scale
        ),
    )
