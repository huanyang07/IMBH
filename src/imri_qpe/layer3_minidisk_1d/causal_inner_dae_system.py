"""Assembled five-field causal Kerr-Schild finite-volume DAE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import splu

from imri_qpe.constants import C, DEFAULT_KAPPA_ES

from .causal_inner_bdf import (
    CausalFiveFieldBDFHistory,
    causal_bdf_coefficients,
    causal_bdf_increment_rate,
)
from .causal_inner_dae import (
    audit_causal_five_field_principal,
    causal_five_field_dae_count,
)
from .causal_inner_geometry import (
    KerrSchildColumnGeometry,
    KerrSchildColumnGrid,
    ValenciaPerfectFluidPrimitive,
    audit_kerr_schild_column_sources,
    kerr_schild_column_geometry,
)
from .causal_inner_migration import (
    KerrSchildCellSourceRates,
    ProperVerticalFrequencyProvider,
    apply_kerr_schild_hill_roche_boundary,
)
from .causal_inner_stress import (
    CausalAlphaShearClosure,
    CausalStressColumnState,
    calibrate_causal_alpha_shear,
    causal_rest_frame_shear_rate,
    causal_stress_column_state,
    causal_stress_relaxation_source,
)
from .causal_inner_thermal import (
    GasRadiationColumnThermodynamics,
    causal_comoving_energy_source,
    causal_temporal_vertical_work_storage,
    causal_thermal_column_source,
    kerr_schild_column_four_velocity,
)
from .hill_roche_nozzle import OverflowBoundaryProvider


_N_FIELDS = 5


@dataclass(frozen=True)
class CausalFiveFieldDAEContext:
    """Immutable physical and numerical inputs for the assembled DAE."""

    grid: KerrSchildColumnGrid
    vertical_frequency: ProperVerticalFrequencyProvider
    outer_boundary_provider: OverflowBoundaryProvider
    stream_sources: KerrSchildCellSourceRates | None = None
    alpha: float = 0.1
    stress_factor: float = 1.0
    kappa: float = DEFAULT_KAPPA_ES
    include_radiative_cooling: bool = True

    def validated(self) -> CausalFiveFieldDAEContext:
        n_cells = int(np.asarray(self.grid.centers).size)
        causal_five_field_dae_count(n_cells)
        if (
            np.asarray(self.grid.edges).shape != (n_cells + 1,)
            or np.asarray(self.grid.cell_measures).shape != (n_cells,)
            or np.asarray(self.grid.face_measures).shape != (n_cells + 1,)
        ):
            raise ValueError("causal DAE grid arrays have inconsistent shapes")
        if not isinstance(
            self.vertical_frequency,
            ProperVerticalFrequencyProvider,
        ):
            raise TypeError("vertical_frequency does not implement its protocol")
        if not isinstance(self.outer_boundary_provider, OverflowBoundaryProvider):
            raise TypeError("outer boundary does not implement its protocol")
        if not np.isclose(
            self.vertical_frequency.gravitational_radius,
            self.grid.gravitational_radius,
            rtol=2.0e-14,
            atol=0.0,
        ):
            raise ValueError("grid and vertical-frequency masses differ")
        if self.stream_sources is not None:
            self.stream_sources.validated_for(n_cells)
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("causal alpha must be positive and finite")
        if not np.isfinite(self.stress_factor) or self.stress_factor <= 0.0:
            raise ValueError("stress_factor must be positive and finite")
        if not np.isfinite(self.kappa) or self.kappa <= 0.0:
            raise ValueError("opacity must be positive and finite")
        return self


@dataclass(frozen=True)
class CausalFiveFieldDAEState:
    """Flux-primary state in conserved, primitive, and weighted-face blocks."""

    conserved: np.ndarray
    primitives: np.ndarray
    weighted_face_fluxes_over_c: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(np.asarray(self.conserved).shape[0])

    def validated(self) -> CausalFiveFieldDAEState:
        n_cells = self.n_cells
        expected = {
            "conserved": (n_cells, _N_FIELDS),
            "primitives": (n_cells, _N_FIELDS),
            "weighted_face_fluxes_over_c": (
                n_cells + 1,
                _N_FIELDS,
            ),
        }
        for name, shape in expected.items():
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != shape or np.any(~np.isfinite(values)):
                raise ValueError(f"{name} has an invalid shape or value")
        causal_five_field_dae_count(n_cells)
        return self


@dataclass(frozen=True)
class CausalFiveFieldCellState:
    """Recovered physical state and closure in one finite-volume cell."""

    geometry: KerrSchildColumnGeometry
    thermodynamics: GasRadiationColumnThermodynamics
    primitive: ValenciaPerfectFluidPrimitive
    closure: CausalAlphaShearClosure
    stress: CausalStressColumnState
    conserved: np.ndarray
    flux_over_c: np.ndarray


@dataclass(frozen=True)
class CausalFiveFieldDAEEvaluation:
    """Assembled residual and the physical blocks used to construct it."""

    residual: np.ndarray
    conservation_rows: np.ndarray
    primitive_map_rows: np.ndarray
    interior_flux_rows: np.ndarray
    inner_flux_rows: np.ndarray
    outer_flux_rows: np.ndarray
    mapped_conserved: np.ndarray
    numerical_weighted_face_fluxes_over_c: np.ndarray
    central_weighted_face_fluxes_over_c: np.ndarray
    rusanov_dissipation_weighted_face_fluxes_over_c: np.ndarray
    integrated_sources_per_ct: np.ndarray
    integrated_source_components_per_ct: dict[str, np.ndarray]
    proper_shear_rates: np.ndarray
    proper_log_height_rates: np.ndarray
    scattering_optical_depths: np.ndarray
    temporal_conserved_storage: np.ndarray
    temporal_vertical_storage: np.ndarray
    outer_boundary_choked: bool
    outer_incoming_characteristics: int

    @property
    def maximum_absolute_residual(self) -> float:
        return float(np.max(np.abs(self.residual)))


@dataclass(frozen=True)
class CausalFiveFieldDAEScaling:
    """Diagonal column and row scales for the physical DAE."""

    column_scales: np.ndarray
    row_scales: np.ndarray

    def validated_for(self, size: int) -> CausalFiveFieldDAEScaling:
        for name in ("column_scales", "row_scales"):
            values = np.asarray(getattr(self, name), dtype=float)
            if (
                values.shape != (size,)
                or np.any(~np.isfinite(values))
                or np.any(values <= 0.0)
            ):
                raise ValueError(f"{name} must be finite and positive")
        return self


@dataclass(frozen=True)
class CausalFiveFieldSparseLinearAudit:
    """Equilibration and residual audit for one sparse Newton solve."""

    dimensions: tuple[int, int]
    nonzeros: int
    row_scale_minimum: float
    row_scale_maximum: float
    column_scale_minimum: float
    column_scale_maximum: float
    relative_linear_residual: float
    method: str


@dataclass(frozen=True)
class CausalFiveFieldJacobianAudit:
    """Dense scaled finite-difference Jacobian rank audit."""

    dimensions: tuple[int, int]
    numerical_rank: int
    singular_values: np.ndarray
    smallest_singular_value: float
    largest_singular_value: float
    condition_estimate: float
    finite_difference_step: float
    scaled_jacobian: np.ndarray
    weakest_right_singular_vector: np.ndarray
    weakest_left_singular_vector: np.ndarray

    @property
    def full_rank(self) -> bool:
        return self.numerical_rank == min(self.dimensions)


@dataclass(frozen=True)
class CausalFiveFieldOuterThermalStressAudit:
    """Two-variable response after eliminating all other primitives."""

    interior_dimensions: tuple[int, int]
    interior_numerical_rank: int
    interior_condition_estimate: float
    response_matrix: np.ndarray
    singular_values: np.ndarray
    numerical_rank: int
    condition_estimate: float
    determinant: float

    @property
    def interior_full_rank(self) -> bool:
        return self.interior_numerical_rank == min(self.interior_dimensions)


@dataclass(frozen=True)
class CausalFiveFieldReducedJacobianAudit:
    """Primitive-only stationary response and full-system Schur comparison."""

    dimensions: tuple[int, int]
    numerical_rank: int
    singular_values: np.ndarray
    smallest_singular_value: float
    largest_singular_value: float
    condition_estimate: float
    finite_difference_step: float
    direct_scaled_jacobian: np.ndarray
    schur_scaled_jacobian: np.ndarray
    schur_singular_values: np.ndarray
    schur_numerical_rank: int
    schur_condition_estimate: float
    maximum_absolute_matrix_defect: float
    relative_frobenius_matrix_defect: float
    maximum_directional_relative_defect: float
    maximum_directional_operator_scaled_defect: float
    algebraic_dimensions: tuple[int, int]
    algebraic_numerical_rank: int
    algebraic_condition_estimate: float
    reconstructed_algebraic_residual_norm: float
    reconstructed_full_residual_norm: float
    full_weakest_vector_alignment: float
    weakest_right_singular_vector: np.ndarray
    weakest_left_singular_vector: np.ndarray
    reconstructed_full_scaled_vector: np.ndarray
    outer_thermal_stress: CausalFiveFieldOuterThermalStressAudit
    outer_boundary_choked: bool

    @property
    def full_rank(self) -> bool:
        return self.numerical_rank == min(self.dimensions)

    @property
    def algebraic_full_rank(self) -> bool:
        return self.algebraic_numerical_rank == min(
            self.algebraic_dimensions
        )


@dataclass(frozen=True)
class CausalFiveFieldConsistentInitialDataAudit:
    """Index-one storage balance and algebraic-tangent compatibility."""

    dimensions: tuple[int, int]
    numerical_rank: int
    singular_values: np.ndarray
    condition_estimate: float
    descriptor_dimensions: tuple[int, int]
    descriptor_numerical_rank: int
    maximum_initial_algebraic_residual: float
    maximum_scaled_consistency_residual: float
    scaled_tangent: np.ndarray
    maximum_scaled_tangent: float
    maximum_scaled_primitive_tangent: float
    storage_balance_residual_norm: float
    algebraic_tangent_residual_norm: float

    @property
    def full_rank(self) -> bool:
        return self.numerical_rank == min(self.dimensions)

    @property
    def descriptor_full_row_rank(self) -> bool:
        return self.descriptor_numerical_rank == self.descriptor_dimensions[0]


@dataclass(frozen=True)
class CausalFiveFieldTemporalStorageIncrement:
    """Finite storage increment along one declared primitive-space path."""

    conserved_increment: np.ndarray
    vertical_killing_increment: np.ndarray
    vertical_work_per_area: np.ndarray
    quadrature_order: int
    directional_step: float
    scheme: str


def pack_causal_five_field_state(
    state: CausalFiveFieldDAEState,
) -> np.ndarray:
    """Pack the exact ``15N+5`` flux-primary state."""

    state = state.validated()
    return np.concatenate(
        (
            np.asarray(state.conserved, dtype=float).ravel(),
            np.asarray(state.primitives, dtype=float).ravel(),
            np.asarray(
                state.weighted_face_fluxes_over_c,
                dtype=float,
            ).ravel(),
        )
    )


def unpack_causal_five_field_state(
    vector: np.ndarray,
    n_cells: int,
) -> CausalFiveFieldDAEState:
    """Unpack one exact ``15N+5`` flux-primary vector."""

    count = causal_five_field_dae_count(n_cells)
    values = np.asarray(vector, dtype=float)
    if values.shape != (count.total_unknowns,) or np.any(~np.isfinite(values)):
        raise ValueError("packed causal five-field state has the wrong shape")
    conserved_end = _N_FIELDS * n_cells
    primitive_end = conserved_end + _N_FIELDS * n_cells
    return CausalFiveFieldDAEState(
        conserved=values[:conserved_end].reshape(n_cells, _N_FIELDS),
        primitives=values[conserved_end:primitive_end].reshape(
            n_cells,
            _N_FIELDS,
        ),
        weighted_face_fluxes_over_c=values[primitive_end:].reshape(
            n_cells + 1,
            _N_FIELDS,
        ),
    ).validated()


def causal_five_field_dae_jacobian_sparsity(
    n_cells: int,
) -> csr_matrix:
    """Return the declared block-local full-DAE Jacobian pattern.

    The ordering matches :func:`pack_causal_five_field_state`: cell-major
    conserved values, cell-major primitive charts, and cell-major face
    fluxes. Conservation sources use one-cell neighbor stencils through the
    shear and responsive-height rates. Primitive maps are cell local, while
    each numerical flux consumes only its adjacent primitive states.
    """

    count = causal_five_field_dae_count(n_cells)
    n_cells = count.n_cells
    conserved_start = 0
    primitive_start = count.conserved_unknowns
    face_start = primitive_start + count.primitive_unknowns
    conservation_start = 0
    primitive_map_start = count.conservation_rows
    interior_flux_start = (
        primitive_map_start + count.primitive_map_rows
    )
    inner_flux_start = (
        interior_flux_start + count.interior_flux_rows
    )
    outer_flux_start = inner_flux_start + count.inner_flux_rows
    pattern = lil_matrix(
        (count.total_rows, count.total_unknowns),
        dtype=np.int8,
    )

    for cell in range(n_cells):
        neighbors = range(
            max(0, cell - 1),
            min(n_cells, cell + 2),
        )
        for component in range(_N_FIELDS):
            conservation_row = (
                conservation_start + _N_FIELDS * cell + component
            )
            pattern[
                conservation_row,
                conserved_start + _N_FIELDS * cell + component,
            ] = 1
            for neighbor in neighbors:
                primitive_slice = slice(
                    primitive_start + _N_FIELDS * neighbor,
                    primitive_start + _N_FIELDS * (neighbor + 1),
                )
                pattern[conservation_row, primitive_slice] = 1
            pattern[
                conservation_row,
                face_start + _N_FIELDS * cell + component,
            ] = 1
            pattern[
                conservation_row,
                face_start + _N_FIELDS * (cell + 1) + component,
            ] = 1

            primitive_map_row = (
                primitive_map_start + _N_FIELDS * cell + component
            )
            pattern[
                primitive_map_row,
                conserved_start + _N_FIELDS * cell + component,
            ] = 1
            primitive_slice = slice(
                primitive_start + _N_FIELDS * cell,
                primitive_start + _N_FIELDS * (cell + 1),
            )
            pattern[primitive_map_row, primitive_slice] = 1

    for face in range(1, n_cells):
        for component in range(_N_FIELDS):
            row = (
                interior_flux_start
                + _N_FIELDS * (face - 1)
                + component
            )
            for cell in (face - 1, face):
                primitive_slice = slice(
                    primitive_start + _N_FIELDS * cell,
                    primitive_start + _N_FIELDS * (cell + 1),
                )
                pattern[row, primitive_slice] = 1
            pattern[
                row,
                face_start + _N_FIELDS * face + component,
            ] = 1

    for component in range(_N_FIELDS):
        inner_row = inner_flux_start + component
        outer_row = outer_flux_start + component
        pattern[
            inner_row,
            primitive_start : primitive_start + _N_FIELDS,
        ] = 1
        pattern[
            inner_row,
            face_start + component,
        ] = 1
        pattern[
            outer_row,
            primitive_start
            + _N_FIELDS * (n_cells - 1) : primitive_start
            + _N_FIELDS * n_cells,
        ] = 1
        pattern[
            outer_row,
            face_start + _N_FIELDS * n_cells + component,
        ] = 1
    return pattern.tocsr()


def causal_five_field_dae_jacobian_color_groups(
    pattern: csr_matrix,
) -> tuple[np.ndarray, ...]:
    """Greedily color columns whose declared residual supports do not meet."""

    declared = pattern.tocsc()
    if declared.shape[0] != declared.shape[1]:
        raise ValueError("causal DAE Jacobian pattern must be square")
    row_colors: list[set[int]] = [
        set() for _ in range(declared.shape[0])
    ]
    groups: list[list[int]] = []
    for column in range(declared.shape[1]):
        start = declared.indptr[column]
        stop = declared.indptr[column + 1]
        rows = declared.indices[start:stop]
        forbidden: set[int] = set()
        for row in rows:
            forbidden.update(row_colors[int(row)])
        color = 0
        while color in forbidden:
            color += 1
        if color == len(groups):
            groups.append([])
        groups[color].append(column)
        for row in rows:
            row_colors[int(row)].add(color)
    return tuple(
        np.asarray(group, dtype=int)
        for group in groups
    )


def causal_five_field_colored_central_jacobian(
    residual,
    values: np.ndarray,
    pattern: csr_matrix,
    *,
    finite_difference_step: float = 2.0e-6,
) -> csr_matrix:
    """Assemble a colored central Jacobian on a certified local pattern."""

    values = np.asarray(values, dtype=float)
    declared = pattern.tocsc()
    if (
        declared.shape != (values.size, values.size)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("colored Jacobian values or pattern are invalid")
    step = float(finite_difference_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("finite-difference step must be positive")
    jacobian = lil_matrix(declared.shape, dtype=float)
    for group in causal_five_field_dae_jacobian_color_groups(declared):
        plus = np.array(values, copy=True)
        minus = np.array(values, copy=True)
        plus[group] += step
        minus[group] -= step
        difference = (
            np.asarray(residual(plus), dtype=float)
            - np.asarray(residual(minus), dtype=float)
        ) / (2.0 * step)
        for column in group:
            start = declared.indptr[column]
            stop = declared.indptr[column + 1]
            rows = declared.indices[start:stop]
            jacobian[rows, column] = difference[rows, None]
    return jacobian.tocsr()


def causal_five_field_equilibrated_sparse_solve(
    matrix: csr_matrix,
    right_hand_side: np.ndarray,
) -> tuple[np.ndarray, CausalFiveFieldSparseLinearAudit]:
    """Solve one scaled Newton system after sparse max-norm equilibration."""

    sparse = matrix.tocsr().astype(float)
    right = np.asarray(right_hand_side, dtype=float)
    if (
        sparse.shape[0] != sparse.shape[1]
        or right.shape != (sparse.shape[0],)
        or np.any(~np.isfinite(sparse.data))
        or np.any(~np.isfinite(right))
    ):
        raise ValueError("sparse Newton system is invalid")
    tiny = np.finfo(float).tiny
    row_maximum = np.asarray(
        np.abs(sparse).max(axis=1).toarray(),
        dtype=float,
    ).ravel()
    if np.any(row_maximum <= tiny):
        raise np.linalg.LinAlgError("sparse Newton matrix has a zero row")
    row_scale = 1.0 / row_maximum
    row_scaled = diags(row_scale) @ sparse
    column_maximum = np.asarray(
        np.abs(row_scaled).max(axis=0).toarray(),
        dtype=float,
    ).ravel()
    if np.any(column_maximum <= tiny):
        raise np.linalg.LinAlgError("sparse Newton matrix has a zero column")
    column_scale = 1.0 / column_maximum
    balanced = (
        diags(row_scale)
        @ sparse
        @ diags(column_scale)
    ).tocsc()
    factor = splu(balanced, permc_spec="COLAMD")
    balanced_solution = factor.solve(row_scale * right)
    solution = column_scale * balanced_solution
    relative_residual = float(
        np.max(np.abs(sparse @ solution - right))
        / max(np.max(np.abs(right)), tiny)
    )
    return solution, CausalFiveFieldSparseLinearAudit(
        dimensions=sparse.shape,
        nonzeros=int(sparse.nnz),
        row_scale_minimum=float(np.min(row_scale)),
        row_scale_maximum=float(np.max(row_scale)),
        column_scale_minimum=float(np.min(column_scale)),
        column_scale_maximum=float(np.max(column_scale)),
        relative_linear_residual=relative_residual,
        method="max_norm_equilibrated_splu_colamd",
    )


def _primitive_from_chart(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
) -> tuple[
    KerrSchildColumnGeometry,
    GasRadiationColumnThermodynamics,
    ValenciaPerfectFluidPrimitive,
]:
    """Recover one responsive column from ``lnSigma,betaR,betaPhi,lnT,chi``."""

    chart = np.asarray(chart, dtype=float)
    if chart.shape != (_N_FIELDS,) or np.any(~np.isfinite(chart)):
        raise ValueError("causal primitive chart must be finite and length five")
    log_sigma, beta_r, beta_phi, log_temperature, _specific_stress = chart
    sigma = float(np.exp(log_sigma))
    temperature = float(np.exp(log_temperature))
    if beta_r**2 + beta_phi**2 >= 1.0:
        raise ValueError("causal primitive velocity is not subluminal")
    geometry = kerr_schild_column_geometry(
        radius,
        context.grid.gravitational_radius,
    )
    eos = context.vertical_frequency.eos(radius)
    thermodynamics = eos.from_surface_density_temperature(
        sigma,
        temperature,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=sigma,
        radial_velocity_over_c=float(beta_r),
        azimuthal_velocity_over_c=float(beta_phi),
        specific_internal_energy=thermodynamics.specific_internal_energy,
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    return geometry, thermodynamics, primitive


def _closure(
    context: CausalFiveFieldDAEContext,
    radius: float,
    thermodynamics: GasRadiationColumnThermodynamics,
    primitive: ValenciaPerfectFluidPrimitive,
) -> CausalAlphaShearClosure:
    """Return the state-local causal alpha calibration."""

    return calibrate_causal_alpha_shear(
        primitive,
        alpha=context.alpha,
        stress_factor=context.stress_factor,
        reference_positive_shear_rate=(
            1.5 * context.vertical_frequency.frequency(radius)
        ),
        viscous_signal_speed_over_c=(
            np.sqrt(context.alpha) * thermodynamics.sound_speed / C
        ),
    )


def _cell_state(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
) -> CausalFiveFieldCellState:
    geometry, thermodynamics, primitive = _primitive_from_chart(
        context,
        radius,
        chart,
    )
    closure = _closure(
        context,
        radius,
        thermodynamics,
        primitive,
    )
    stress = causal_stress_column_state(
        geometry,
        primitive,
        specific_stress=float(chart[4]),
    )
    conserved = np.concatenate(
        (
            stress.killing_conserved,
            [stress.relaxing_stress_conserved],
        )
    )
    flux = np.concatenate(
        (
            stress.killing_flux_over_c,
            [stress.relaxing_stress_flux_over_c],
        )
    )
    return CausalFiveFieldCellState(
        geometry=geometry,
        thermodynamics=thermodynamics,
        primitive=primitive,
        closure=closure,
        stress=stress,
        conserved=np.asarray(conserved, dtype=float),
        flux_over_c=np.asarray(flux, dtype=float),
    )


def causal_five_field_cell_states(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
) -> tuple[CausalFiveFieldCellState, ...]:
    """Recover the physical cell states from one packed DAE vector."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(vector, n_cells)
    return tuple(
        _cell_state(context, float(radius), chart)
        for radius, chart in zip(
            context.grid.centers,
            state.primitives,
            strict=True,
        )
    )


def _interior_rusanov_flux_components(
    context: CausalFiveFieldDAEContext,
    face_index: int,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return central and dissipative pieces of one Rusanov flux."""

    radius = float(context.grid.edges[face_index])
    left = _cell_state(context, radius, left_chart)
    right = _cell_state(context, radius, right_chart)
    speeds = []
    for state in (left, right):
        audit = audit_causal_five_field_principal(
            state.geometry,
            context.vertical_frequency.eos(radius),
            state.closure,
            surface_density=state.primitive.surface_density,
            radial_velocity_over_c=(
                state.primitive.radial_velocity_over_c
            ),
            azimuthal_velocity_over_c=(
                state.primitive.azimuthal_velocity_over_c
            ),
            temperature=state.thermodynamics.temperature,
        )
        speeds.extend(audit.coordinate_speeds_over_c)
    maximum_speed = float(np.max(np.abs(speeds)))
    measure = float(context.grid.face_measures[face_index])
    central = (
        measure * 0.5 * (left.flux_over_c + right.flux_over_c)
    )
    dissipation = (
        -measure
        * 0.5
        * maximum_speed
        * (right.conserved - left.conserved)
    )
    return (
        np.asarray(central, dtype=float),
        np.asarray(dissipation, dtype=float),
    )


def _interior_rusanov_flux(
    context: CausalFiveFieldDAEContext,
    face_index: int,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
) -> np.ndarray:
    """Return one proper-measure weighted five-field Rusanov flux."""

    central, dissipation = _interior_rusanov_flux_components(
        context,
        face_index,
        left_chart,
        right_chart,
    )
    return central + dissipation


def _inner_face_flux(
    context: CausalFiveFieldDAEContext,
    chart: np.ndarray,
) -> np.ndarray:
    """Return the one-sided excision flux; no physical inner BC is imposed."""

    state = _cell_state(
        context,
        float(context.grid.edges[0]),
        chart,
    )
    return np.asarray(
        context.grid.face_measures[0] * state.flux_over_c,
        dtype=float,
    )


def _outer_face_flux(
    context: CausalFiveFieldDAEContext,
    chart: np.ndarray,
) -> tuple[np.ndarray, bool, int]:
    """Return the physical Roche acoustic flux plus zero shear stress."""

    radius = float(context.grid.edges[-1])
    geometry, thermodynamics, primitive = _primitive_from_chart(
        context,
        radius,
        chart,
    )
    boundary = apply_kerr_schild_hill_roche_boundary(
        geometry,
        context.vertical_frequency.eos(radius),
        primitive,
        temperature=thermodynamics.temperature,
        provider=context.outer_boundary_provider,
        outer_specific_stress=0.0,
    )
    return (
        np.concatenate(
            (
                boundary.weighted_killing_flux_over_c,
                [0.0],
            )
        ),
        bool(boundary.gate.choked),
        int(boundary.incoming_outer_characteristics + 1),
    )


def _straight_path_cell_rates(
    context: CausalFiveFieldDAEContext,
    cell_states: list[CausalFiveFieldCellState],
) -> tuple[np.ndarray, np.ndarray]:
    """Return covariant shear and radial height rates on one declared path."""

    n_cells = len(cell_states)
    lower_velocity = np.asarray(
        [
            state.geometry.spacetime_metric
            @ kerr_schild_column_four_velocity(
                state.geometry,
                state.primitive,
            )
            for state in cell_states
        ],
        dtype=float,
    )
    face_lower_velocity = np.empty((n_cells + 1, 3), dtype=float)
    face_lower_velocity[0] = lower_velocity[0]
    face_lower_velocity[-1] = lower_velocity[-1]
    if n_cells > 1:
        face_lower_velocity[1:-1] = 0.5 * (
            lower_velocity[:-1] + lower_velocity[1:]
        )

    log_height = np.log(
        [
            state.thermodynamics.proper_half_thickness
            for state in cell_states
        ]
    )
    face_log_height = np.empty(n_cells + 1, dtype=float)
    face_log_height[0] = log_height[0]
    face_log_height[-1] = log_height[-1]
    if n_cells > 1:
        face_log_height[1:-1] = 0.5 * (
            log_height[:-1] + log_height[1:]
        )

    widths = np.diff(context.grid.edges)
    shear = np.empty(n_cells, dtype=float)
    height_rate = np.empty(n_cells, dtype=float)
    for index, state in enumerate(cell_states):
        derivative = (
            face_lower_velocity[index + 1]
            - face_lower_velocity[index]
        ) / widths[index]
        shear[index] = causal_rest_frame_shear_rate(
            state.geometry,
            state.primitive,
            radial_lower_four_velocity_derivative=derivative,
        )
        log_height_derivative = (
            face_log_height[index + 1] - face_log_height[index]
        ) / widths[index]
        four_velocity = kerr_schild_column_four_velocity(
            state.geometry,
            state.primitive,
        )
        height_rate[index] = (
            C * four_velocity[1] * log_height_derivative
        )
    return shear, height_rate


def _integrated_cell_sources(
    context: CausalFiveFieldDAEContext,
    cell_states: list[CausalFiveFieldCellState],
    shear_rates: np.ndarray,
    height_rates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return cell-integrated five-field sources per coordinate ``ct``."""

    n_cells = len(cell_states)
    sources = np.zeros((n_cells, _N_FIELDS), dtype=float)
    components = {
        name: np.zeros((n_cells, _N_FIELDS), dtype=float)
        for name in (
            "perfect_fluid_geometry",
            "stress_geometry",
            "radiative_cooling",
            "vertical_work",
            "stress_relaxation",
            "stream",
        )
    }
    optical_depths = np.full(n_cells, np.nan, dtype=float)
    for index, state in enumerate(cell_states):
        perfect = audit_kerr_schild_column_sources(
            state.geometry,
            state.primitive,
        )
        measure = float(context.grid.cell_measures[index])
        components["perfect_fluid_geometry"][index, 1] = (
            measure * perfect.radial_momentum_source
        )
        components["stress_geometry"][index, 1] = (
            measure * state.stress.radial_geometric_source_increment
        )
        local = np.asarray(
            [
                0.0,
                (
                    perfect.radial_momentum_source
                    + state.stress.radial_geometric_source_increment
                ),
                0.0,
                0.0,
            ],
            dtype=float,
        )
        if context.include_radiative_cooling:
            thermal = causal_thermal_column_source(
                state.geometry,
                context.vertical_frequency.eos(state.geometry.radius),
                surface_density=state.primitive.surface_density,
                radial_velocity_over_c=(
                    state.primitive.radial_velocity_over_c
                ),
                azimuthal_velocity_over_c=(
                    state.primitive.azimuthal_velocity_over_c
                ),
                temperature=state.thermodynamics.temperature,
                proper_log_height_rate=float(height_rates[index]),
                kappa=context.kappa,
            )
            local += thermal.total_killing_source_per_ct
            components["radiative_cooling"][index, :4] = (
                measure
                * thermal.cooling_source.killing_source_per_ct
            )
            components["vertical_work"][index, :4] = (
                measure
                * thermal.vertical_work_source.killing_source_per_ct
            )
            optical_depths[index] = thermal.scattering_optical_depth
        else:
            vertical_work = (
                -state.thermodynamics.integrated_pressure
                * height_rates[index]
            )
            vertical_source = causal_comoving_energy_source(
                state.geometry,
                state.primitive,
                comoving_energy_rate=float(vertical_work),
            )
            local += vertical_source.killing_source_per_ct
            components["vertical_work"][index, :4] = (
                measure * vertical_source.killing_source_per_ct
            )
            optical_depths[index] = (
                0.5
                * context.kappa
                * state.thermodynamics.surface_density
            )

        sources[index, :4] = measure * local
        stress_relaxation = causal_stress_relaxation_source(
            state.geometry,
            state.stress,
            state.closure,
            positive_shear_rate=float(shear_rates[index]),
        )
        sources[index, 4] = measure * stress_relaxation
        components["stress_relaxation"][index, 4] = (
            measure * stress_relaxation
        )
    if context.stream_sources is not None:
        stream = np.asarray(
            context.stream_sources.weighted_killing_source_per_ct,
            dtype=float,
        )
        sources[:, :4] += stream
        components["stream"][:, :4] = stream
    return sources, optical_depths, components


def _mapped_state_and_fluxes(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
) -> tuple[
    list[CausalFiveFieldCellState],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
    int,
]:
    """Map primitive charts to cell storage and numerical face fluxes."""

    cell_states = [
        _cell_state(context, float(radius), chart)
        for radius, chart in zip(
            context.grid.centers,
            primitive_charts,
            strict=True,
        )
    ]
    mapped = np.asarray(
        [state.conserved for state in cell_states],
        dtype=float,
    )
    n_cells = len(cell_states)
    faces = np.empty((n_cells + 1, _N_FIELDS), dtype=float)
    central_faces = np.empty_like(faces)
    dissipative_faces = np.zeros_like(faces)
    faces[0] = _inner_face_flux(context, primitive_charts[0])
    central_faces[0] = faces[0]
    for face in range(1, n_cells):
        central_faces[face], dissipative_faces[face] = (
            _interior_rusanov_flux_components(
                context,
                face,
                primitive_charts[face - 1],
                primitive_charts[face],
            )
        )
        faces[face] = (
            central_faces[face] + dissipative_faces[face]
        )
    faces[-1], choked, incoming = _outer_face_flux(
        context,
        primitive_charts[-1],
    )
    central_faces[-1] = faces[-1]
    return (
        cell_states,
        mapped,
        faces,
        central_faces,
        dissipative_faces,
        choked,
        incoming,
    )


def evaluate_causal_five_field_dae(
    vector: np.ndarray,
    context: CausalFiveFieldDAEContext,
    *,
    old_vector: np.ndarray | None = None,
    timestep_seconds: float | None = None,
    temporal_storage_scheme: str = "endpoint",
) -> CausalFiveFieldDAEEvaluation:
    """Evaluate the stationary or backward-Euler flux-primary residual."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(vector, n_cells)
    if (old_vector is None) != (timestep_seconds is None):
        raise ValueError("old state and timestep must be supplied together")
    if timestep_seconds is not None and (
        not np.isfinite(timestep_seconds) or timestep_seconds <= 0.0
    ):
        raise ValueError("backward-Euler timestep must be positive and finite")
    if temporal_storage_scheme not in ("endpoint", "path_integrated"):
        raise ValueError("unknown temporal-storage scheme")

    (
        cell_states,
        mapped,
        numerical_fluxes,
        central_fluxes,
        dissipative_fluxes,
        choked,
        incoming,
    ) = _mapped_state_and_fluxes(context, state.primitives)
    shear_rates, height_rates = _straight_path_cell_rates(
        context,
        cell_states,
    )
    sources, optical_depths, source_components = _integrated_cell_sources(
        context,
        cell_states,
        shear_rates,
        height_rates,
    )

    conservation = (
        state.weighted_face_fluxes_over_c[1:]
        - state.weighted_face_fluxes_over_c[:-1]
        - sources
    )
    temporal_conserved_storage = np.zeros(
        (n_cells, _N_FIELDS),
        dtype=float,
    )
    temporal_storage = np.zeros((n_cells, 4), dtype=float)
    if old_vector is not None:
        assert timestep_seconds is not None
        old = unpack_causal_five_field_state(old_vector, n_cells)
        coordinate_timestep = C * timestep_seconds
        if temporal_storage_scheme == "endpoint":
            conserved_increment = state.conserved - old.conserved
            vertical_increment = np.zeros((n_cells, 4), dtype=float)
            for index, (cell_state, old_chart) in enumerate(
                zip(cell_states, old.primitives, strict=True)
            ):
                _old_geometry, old_thermodynamics, _old_primitive = (
                    _primitive_from_chart(
                        context,
                        float(context.grid.centers[index]),
                        old_chart,
                    )
                )
                storage = causal_temporal_vertical_work_storage(
                    cell_state.geometry,
                    cell_state.primitive,
                    old_thermodynamics,
                    cell_state.thermodynamics,
                )
                vertical_increment[index] = (
                    storage.killing_storage_increment
                )
        else:
            old_mapped = np.asarray(
                [
                    _cell_state(
                        context,
                        float(radius),
                        chart,
                    ).conserved
                    for radius, chart in zip(
                        context.grid.centers,
                        old.primitives,
                        strict=True,
                    )
                ],
                dtype=float,
            )
            old_scale = np.maximum(np.abs(old_mapped), 1.0)
            new_scale = np.maximum(np.abs(mapped), 1.0)
            if (
                np.max(
                    np.abs(old.conserved - old_mapped) / old_scale
                )
                > 1.0e-12
                or np.max(
                    np.abs(state.conserved - mapped) / new_scale
                )
                > 1.0e-12
            ):
                raise ValueError(
                    "path-integrated storage requires exact primitive maps"
                )
            path = causal_five_field_path_temporal_storage_increment(
                context,
                old.primitives,
                state.primitives,
            )
            conserved_increment = path.conserved_increment
            vertical_increment = path.vertical_killing_increment
        temporal_conserved_storage = (
            context.grid.cell_measures[:, None]
            * conserved_increment
            / coordinate_timestep
        )
        conservation += temporal_conserved_storage
        temporal_storage = (
            context.grid.cell_measures[:, None]
            * vertical_increment
            / coordinate_timestep
        )
        conservation[:, :4] += temporal_storage

    primitive_map = state.conserved - mapped
    interior_flux = (
        state.weighted_face_fluxes_over_c[1:-1]
        - numerical_fluxes[1:-1]
    )
    inner_flux = state.weighted_face_fluxes_over_c[0] - numerical_fluxes[0]
    outer_flux = state.weighted_face_fluxes_over_c[-1] - numerical_fluxes[-1]
    residual = np.concatenate(
        (
            conservation.ravel(),
            primitive_map.ravel(),
            interior_flux.ravel(),
            inner_flux,
            outer_flux,
        )
    )
    expected = causal_five_field_dae_count(n_cells).total_rows
    if residual.shape != (expected,) or np.any(~np.isfinite(residual)):
        raise ValueError("assembled causal DAE residual is invalid")
    return CausalFiveFieldDAEEvaluation(
        residual=residual,
        conservation_rows=conservation,
        primitive_map_rows=primitive_map,
        interior_flux_rows=interior_flux,
        inner_flux_rows=inner_flux,
        outer_flux_rows=outer_flux,
        mapped_conserved=mapped,
        numerical_weighted_face_fluxes_over_c=numerical_fluxes,
        central_weighted_face_fluxes_over_c=central_fluxes,
        rusanov_dissipation_weighted_face_fluxes_over_c=(
            dissipative_fluxes
        ),
        integrated_sources_per_ct=sources,
        integrated_source_components_per_ct=source_components,
        proper_shear_rates=shear_rates,
        proper_log_height_rates=height_rates,
        scattering_optical_depths=optical_depths,
        temporal_conserved_storage=temporal_conserved_storage,
        temporal_vertical_storage=temporal_storage,
        outer_boundary_choked=choked,
        outer_incoming_characteristics=incoming,
    )


def causal_five_field_bdf_history(
    context: CausalFiveFieldDAEContext,
    current_vector: np.ndarray,
    previous_physical_increment: np.ndarray,
    previous_timestep_seconds: float,
    *,
    temporal_height_scheme: str = "path_integrated",
) -> CausalFiveFieldBDFHistory:
    """Build complete fixed history for the next BDF2 residual."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    count = causal_five_field_dae_count(n_cells)
    current_values = np.asarray(current_vector, dtype=float)
    previous_increment = np.asarray(
        previous_physical_increment,
        dtype=float,
    )
    if (
        current_values.shape != (count.total_unknowns,)
        or previous_increment.shape != current_values.shape
        or np.any(~np.isfinite(current_values))
        or np.any(~np.isfinite(previous_increment))
    ):
        raise ValueError("causal BDF history vectors are invalid")
    previous_timestep = float(previous_timestep_seconds)
    if not np.isfinite(previous_timestep) or previous_timestep <= 0.0:
        raise ValueError("previous BDF timestep must be positive")
    if temporal_height_scheme not in ("endpoint", "path_integrated"):
        raise ValueError("unknown temporal-height scheme")
    current = unpack_causal_five_field_state(
        current_values,
        n_cells,
    )
    previous = unpack_causal_five_field_state(
        current_values - previous_increment,
        n_cells,
    )
    if temporal_height_scheme == "endpoint":
        temporal = causal_five_field_endpoint_temporal_storage_increment(
            context,
            previous.primitives,
            current.primitives,
        )
    else:
        temporal = causal_five_field_path_temporal_storage_increment(
            context,
            previous.primitives,
            current.primitives,
        )
    return CausalFiveFieldBDFHistory(
        previous_physical_increment=previous_increment,
        previous_vertical_killing_increment=np.asarray(
            temporal.vertical_killing_increment,
            dtype=float,
        ),
        previous_timestep_seconds=previous_timestep,
        temporal_height_scheme=temporal_height_scheme,
    ).validated(
        total_unknowns=count.total_unknowns,
        n_cells=n_cells,
    )


def evaluate_causal_five_field_increment_bdf(
    increment_vector: np.ndarray,
    context: CausalFiveFieldDAEContext,
    *,
    old_vector: np.ndarray,
    timestep_seconds: float,
    order: int,
    history: CausalFiveFieldBDFHistory | None = None,
    temporal_height_scheme: str = "path_integrated",
) -> CausalFiveFieldDAEEvaluation:
    """Evaluate BDF1 or BDF2 with primary state and face increments.

    The conserved increment is an independent Newton unknown and enters the
    amplified storage row directly. Primitive recovery and numerical face
    closure remain algebraic constraints at the new state.
    """

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    count = causal_five_field_dae_count(n_cells)
    increment = np.asarray(increment_vector, dtype=float)
    old_values = np.asarray(old_vector, dtype=float)
    if (
        increment.shape != (count.total_unknowns,)
        or old_values.shape != (count.total_unknowns,)
        or np.any(~np.isfinite(increment))
        or np.any(~np.isfinite(old_values))
    ):
        raise ValueError("increment-primary DAE vectors are invalid")
    timestep = float(timestep_seconds)
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("BDF timestep must be positive and finite")
    if temporal_height_scheme not in ("endpoint", "path_integrated"):
        raise ValueError("unknown temporal-height scheme")
    if int(order) != order or order not in (1, 2):
        raise ValueError("causal BDF order must be one or two")
    if order == 1:
        if history is not None:
            raise ValueError("BDF1 does not consume temporal history")
        validated_history = None
        coefficients = causal_bdf_coefficients(1, timestep)
    else:
        if history is None:
            raise ValueError("BDF2 requires complete temporal history")
        validated_history = history.validated(
            total_unknowns=count.total_unknowns,
            n_cells=n_cells,
        )
        if (
            validated_history.temporal_height_scheme
            != temporal_height_scheme
        ):
            raise ValueError(
                "BDF history uses a different temporal-height scheme"
            )
        coefficients = causal_bdf_coefficients(
            order,
            timestep,
            validated_history.previous_timestep_seconds,
        )

    old = unpack_causal_five_field_state(old_values, n_cells)
    new_state = unpack_causal_five_field_state(
        old_values + increment,
        n_cells,
    )
    stationary = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(new_state),
        context,
    )
    n_differential = _N_FIELDS * n_cells
    conserved_increment = increment[:n_differential].reshape(
        n_cells,
        _N_FIELDS,
    )
    if temporal_height_scheme == "endpoint":
        temporal = causal_five_field_endpoint_temporal_storage_increment(
            context,
            old.primitives,
            new_state.primitives,
        )
    else:
        temporal = causal_five_field_path_temporal_storage_increment(
            context,
            old.primitives,
            new_state.primitives,
        )
    previous_conserved_increment = (
        None
        if validated_history is None
        else validated_history.previous_physical_increment[
            :n_differential
        ].reshape(n_cells, _N_FIELDS)
    )
    previous_vertical_increment = (
        None
        if validated_history is None
        else validated_history.previous_vertical_killing_increment
    )
    temporal_conserved_storage = (
        context.grid.cell_measures[:, None]
        * causal_bdf_increment_rate(
            conserved_increment,
            previous_conserved_increment,
            coefficients,
        )
        / C
    )
    temporal_vertical_storage = (
        context.grid.cell_measures[:, None]
        * causal_bdf_increment_rate(
            temporal.vertical_killing_increment,
            previous_vertical_increment,
            coefficients,
        )
        / C
    )
    conservation = (
        stationary.conservation_rows
        + temporal_conserved_storage
    )
    conservation[:, :4] += temporal_vertical_storage
    residual = np.concatenate(
        (
            conservation.ravel(),
            stationary.primitive_map_rows.ravel(),
            stationary.interior_flux_rows.ravel(),
            stationary.inner_flux_rows,
            stationary.outer_flux_rows,
        )
    )
    if residual.shape != (count.total_rows,) or np.any(~np.isfinite(residual)):
        raise ValueError("increment-primary DAE residual is invalid")
    return CausalFiveFieldDAEEvaluation(
        residual=residual,
        conservation_rows=conservation,
        primitive_map_rows=stationary.primitive_map_rows,
        interior_flux_rows=stationary.interior_flux_rows,
        inner_flux_rows=stationary.inner_flux_rows,
        outer_flux_rows=stationary.outer_flux_rows,
        mapped_conserved=stationary.mapped_conserved,
        numerical_weighted_face_fluxes_over_c=(
            stationary.numerical_weighted_face_fluxes_over_c
        ),
        central_weighted_face_fluxes_over_c=(
            stationary.central_weighted_face_fluxes_over_c
        ),
        rusanov_dissipation_weighted_face_fluxes_over_c=(
            stationary.rusanov_dissipation_weighted_face_fluxes_over_c
        ),
        integrated_sources_per_ct=stationary.integrated_sources_per_ct,
        integrated_source_components_per_ct=(
            stationary.integrated_source_components_per_ct
        ),
        proper_shear_rates=stationary.proper_shear_rates,
        proper_log_height_rates=stationary.proper_log_height_rates,
        scattering_optical_depths=stationary.scattering_optical_depths,
        temporal_conserved_storage=temporal_conserved_storage,
        temporal_vertical_storage=temporal_vertical_storage,
        outer_boundary_choked=stationary.outer_boundary_choked,
        outer_incoming_characteristics=(
            stationary.outer_incoming_characteristics
        ),
    )


def evaluate_causal_five_field_increment_backward_euler(
    increment_vector: np.ndarray,
    context: CausalFiveFieldDAEContext,
    *,
    old_vector: np.ndarray,
    timestep_seconds: float,
    temporal_height_scheme: str = "path_integrated",
) -> CausalFiveFieldDAEEvaluation:
    """Evaluate the order-one increment-primary BDF formula."""

    return evaluate_causal_five_field_increment_bdf(
        increment_vector,
        context,
        old_vector=old_vector,
        timestep_seconds=timestep_seconds,
        order=1,
        temporal_height_scheme=temporal_height_scheme,
    )


def causal_five_field_state_from_primitives(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
) -> CausalFiveFieldDAEState:
    """Create a flux-consistent state from one primitive chart per cell."""

    context = context.validated()
    primitives = np.asarray(primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    if primitives.shape != (n_cells, _N_FIELDS):
        raise ValueError("primitive seed has the wrong shape")
    (
        _states,
        mapped,
        faces,
        _central_faces,
        _dissipative_faces,
        _choked,
        _incoming,
    ) = _mapped_state_and_fluxes(context, primitives)
    return CausalFiveFieldDAEState(
        conserved=mapped,
        primitives=np.array(primitives, copy=True),
        weighted_face_fluxes_over_c=faces,
    ).validated()


def causal_five_field_reduced_stationary_residual(
    primitive_vector: np.ndarray,
    context: CausalFiveFieldDAEContext,
) -> np.ndarray:
    """Return conservation rows after exact primitive/face elimination."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    primitives = np.asarray(primitive_vector, dtype=float)
    if (
        primitives.shape != (_N_FIELDS * n_cells,)
        or np.any(~np.isfinite(primitives))
    ):
        raise ValueError("reduced primitive vector has the wrong shape or value")
    state = causal_five_field_state_from_primitives(
        context,
        primitives.reshape(n_cells, _N_FIELDS),
    )
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    return np.asarray(evaluation.conservation_rows, dtype=float).ravel()


def causal_five_field_reduced_backward_euler_residual(
    primitive_vector: np.ndarray,
    context: CausalFiveFieldDAEContext,
    *,
    old_vector: np.ndarray,
    timestep_seconds: float,
    temporal_storage_scheme: str = "endpoint",
) -> np.ndarray:
    """Return backward-Euler conservation after exact map elimination."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    primitives = np.asarray(primitive_vector, dtype=float)
    if (
        primitives.shape != (_N_FIELDS * n_cells,)
        or np.any(~np.isfinite(primitives))
    ):
        raise ValueError("reduced primitive vector has the wrong shape or value")
    old = np.asarray(old_vector, dtype=float)
    expected = causal_five_field_dae_count(n_cells).total_unknowns
    if old.shape != (expected,) or np.any(~np.isfinite(old)):
        raise ValueError("old DAE state has the wrong shape or value")
    state = causal_five_field_state_from_primitives(
        context,
        primitives.reshape(n_cells, _N_FIELDS),
    )
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
        old_vector=old,
        timestep_seconds=timestep_seconds,
        temporal_storage_scheme=temporal_storage_scheme,
    )
    return np.asarray(evaluation.conservation_rows, dtype=float).ravel()


def _vertical_storage_increment_from_work(
    geometry: KerrSchildColumnGeometry,
    primitive: ValenciaPerfectFluidPrimitive,
    work_per_area: float,
) -> np.ndarray:
    four_velocity = kerr_schild_column_four_velocity(
        geometry,
        primitive,
    )
    lower_velocity = geometry.spacetime_metric @ four_velocity
    coefficient = (
        geometry.base.lapse
        * float(work_per_area)
        * four_velocity[0]
        / C**2
    )
    return np.asarray(
        [
            0.0,
            coefficient * lower_velocity[1],
            coefficient * lower_velocity[2],
            -coefficient * lower_velocity[0],
        ],
        dtype=float,
    )


def causal_five_field_endpoint_temporal_storage_increment(
    context: CausalFiveFieldDAEContext,
    old_primitive_charts: np.ndarray,
    new_primitive_charts: np.ndarray,
) -> CausalFiveFieldTemporalStorageIncrement:
    """Return the original endpoint/trapezoidal finite storage increment."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    old = np.asarray(old_primitive_charts, dtype=float)
    new = np.asarray(new_primitive_charts, dtype=float)
    if (
        old.shape != (n_cells, _N_FIELDS)
        or new.shape != (n_cells, _N_FIELDS)
        or np.any(~np.isfinite(old))
        or np.any(~np.isfinite(new))
    ):
        raise ValueError("temporal-storage primitive arrays are invalid")
    old_states = [
        _cell_state(context, float(radius), chart)
        for radius, chart in zip(context.grid.centers, old, strict=True)
    ]
    new_states = [
        _cell_state(context, float(radius), chart)
        for radius, chart in zip(context.grid.centers, new, strict=True)
    ]
    conserved_increment = np.asarray(
        [
            new_state.conserved - old_state.conserved
            for old_state, new_state in zip(
                old_states,
                new_states,
                strict=True,
            )
        ],
        dtype=float,
    )
    vertical_increment = np.zeros((n_cells, 4), dtype=float)
    work = np.zeros(n_cells, dtype=float)
    for index, (old_state, new_state) in enumerate(
        zip(old_states, new_states, strict=True)
    ):
        storage = causal_temporal_vertical_work_storage(
            new_state.geometry,
            new_state.primitive,
            old_state.thermodynamics,
            new_state.thermodynamics,
        )
        work[index] = storage.work_per_area
        vertical_increment[index] = storage.killing_storage_increment
    return CausalFiveFieldTemporalStorageIncrement(
        conserved_increment=conserved_increment,
        vertical_killing_increment=vertical_increment,
        vertical_work_per_area=work,
        quadrature_order=0,
        directional_step=0.0,
        scheme="endpoint",
    )


def causal_five_field_path_temporal_storage_increment(
    context: CausalFiveFieldDAEContext,
    old_primitive_charts: np.ndarray,
    new_primitive_charts: np.ndarray,
    *,
    quadrature_order: int = 2,
    directional_step: float = 1.0e-3,
) -> CausalFiveFieldTemporalStorageIncrement:
    """Integrate finite storage on a straight primitive-space path.

    Fourth-order centered coordinate derivatives avoid subtracting endpoint
    conserved states at the timestep-sized separation. Their Jacobian-vector
    product uses the primitive endpoint increment directly, without a
    magnitude/direction normalization that becomes noisy under tiny Newton
    corrections. Gauss-Legendre quadrature then integrates both the exact-state
    derivative and the responsive-height ``Pi dlnH`` one-form along the same
    declared path.
    """

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    old = np.asarray(old_primitive_charts, dtype=float)
    new = np.asarray(new_primitive_charts, dtype=float)
    if (
        old.shape != (n_cells, _N_FIELDS)
        or new.shape != (n_cells, _N_FIELDS)
        or np.any(~np.isfinite(old))
        or np.any(~np.isfinite(new))
    ):
        raise ValueError("temporal-storage primitive arrays are invalid")
    order = int(quadrature_order)
    if order != quadrature_order or not 2 <= order <= 16:
        raise ValueError("storage quadrature order must lie in [2, 16]")
    step = float(directional_step)
    if not np.isfinite(step) or not 1.0e-5 <= step <= 5.0e-3:
        raise ValueError("storage directional step lies outside its audit range")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    lambdas = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    conserved_increment = np.zeros((n_cells, _N_FIELDS), dtype=float)
    vertical_increment = np.zeros((n_cells, 4), dtype=float)
    work_increment = np.zeros(n_cells, dtype=float)

    for cell, radius in enumerate(context.grid.centers):
        delta = new[cell] - old[cell]
        primitive_scale = np.ones(_N_FIELDS, dtype=float)
        primitive_scale[4] = max(abs(old[cell, 4]), 1.0e-14)
        normalized_delta = delta / primitive_scale
        if not np.any(normalized_delta):
            continue

        for path_fraction, weight in zip(
            lambdas,
            weights,
            strict=True,
        ):
            center = old[cell] + path_fraction * delta
            derivative_weights = np.asarray(
                [1.0, -8.0, 8.0, -1.0],
                dtype=float,
            ) / (12.0 * step)
            conserved_derivative = np.zeros(_N_FIELDS, dtype=float)
            log_height_derivative = 0.0
            for field in range(_N_FIELDS):
                if normalized_delta[field] == 0.0:
                    continue
                perturbation = np.zeros(_N_FIELDS, dtype=float)
                perturbation[field] = step * primitive_scale[field]
                samples = [
                    _cell_state(
                        context,
                        float(radius),
                        center + multiplier * perturbation,
                    )
                    for multiplier in (-2.0, -1.0, 1.0, 2.0)
                ]
                conserved_coordinate_derivative = np.sum(
                    derivative_weights[:, None]
                    * np.asarray(
                        [sample.conserved for sample in samples],
                        dtype=float,
                    ),
                    axis=0,
                )
                log_height_coordinate_derivative = float(
                    np.dot(
                        derivative_weights,
                        np.log(
                            [
                                sample.thermodynamics.proper_half_thickness
                                for sample in samples
                            ]
                        ),
                    )
                )
                conserved_derivative += (
                    normalized_delta[field]
                    * conserved_coordinate_derivative
                )
                log_height_derivative += (
                    normalized_delta[field]
                    * log_height_coordinate_derivative
                )
            center_state = _cell_state(
                context,
                float(radius),
                center,
            )
            work_rate = (
                center_state.thermodynamics.integrated_pressure
                * log_height_derivative
            )
            conserved_increment[cell] += (
                weight * conserved_derivative
            )
            work_increment[cell] += weight * work_rate
            vertical_increment[cell] += (
                weight
                * _vertical_storage_increment_from_work(
                    center_state.geometry,
                    center_state.primitive,
                    work_rate,
                )
            )
    return CausalFiveFieldTemporalStorageIncrement(
        conserved_increment=conserved_increment,
        vertical_killing_increment=vertical_increment,
        vertical_work_per_area=work_increment,
        quadrature_order=order,
        directional_step=step,
        scheme="path_integrated",
    )


def make_causal_five_field_seed(
    context: CausalFiveFieldDAEContext,
    *,
    inner_surface_density: float = 1.0e7,
    outer_surface_density: float = 1.0e5,
    inner_temperature: float = 3.0e7,
    outer_temperature: float = 8.0e5,
    inner_radial_velocity_over_c: float = -0.40,
    inner_azimuthal_velocity_over_c: float = 0.60,
    outer_radial_velocity_margin_over_c: float = 1.0e-5,
    profile_inner_plateau_radius: float | None = None,
    profile_outer_plateau_radius: float | None = None,
    profile_interpolate_log_h_over_r: bool = False,
) -> CausalFiveFieldDAEState:
    """Return a smooth low-throughput, alpha-equilibrium preflight seed.

    When both plateau radii are supplied, the primitive endpoints are fixed
    physical states joined by one C2 smootherstep in log radius. This gives
    different meshes samples of the same continuum profile. With
    ``profile_interpolate_log_h_over_r``, the endpoint temperatures instead
    define H/R at the physical domain faces; local temperatures are recovered
    from a C2 log-H/R profile. This avoids a radiation-pressure thickness
    bulge from independently interpolating surface density and temperature.
    """

    context = context.validated()
    radius = np.asarray(context.grid.centers, dtype=float)
    if (
        profile_inner_plateau_radius is None
        and profile_outer_plateau_radius is None
    ):
        fraction = (
            np.log(radius / radius[0])
            / np.log(radius[-1] / radius[0])
            if radius.size > 1
            else np.zeros(1)
        )
    elif (
        profile_inner_plateau_radius is None
        or profile_outer_plateau_radius is None
    ):
        raise ValueError("both profile plateau radii are required")
    else:
        inner_plateau = float(profile_inner_plateau_radius)
        outer_plateau = float(profile_outer_plateau_radius)
        if (
            not np.isfinite(inner_plateau)
            or not np.isfinite(outer_plateau)
            or inner_plateau < context.grid.edges[0]
            or outer_plateau > context.grid.edges[-1]
            or outer_plateau <= inner_plateau
        ):
            raise ValueError("profile plateau radii are invalid")
        coordinate = np.clip(
            (
                np.log(radius / inner_plateau)
                / np.log(outer_plateau / inner_plateau)
            ),
            0.0,
            1.0,
        )
        fraction = (
            coordinate**3
            * (
                10.0
                - 15.0 * coordinate
                + 6.0 * coordinate**2
            )
        )
    sigma = np.exp(
        (1.0 - fraction) * np.log(inner_surface_density)
        + fraction * np.log(outer_surface_density)
    )
    if profile_interpolate_log_h_over_r:
        if (
            profile_inner_plateau_radius is None
            or profile_outer_plateau_radius is None
        ):
            raise ValueError(
                "log-H/R interpolation requires fixed profile anchors"
            )
        inner_radius = float(context.grid.edges[0])
        outer_radius = float(context.grid.edges[-1])
        inner_eos = context.vertical_frequency.eos(inner_radius)
        outer_eos = context.vertical_frequency.eos(outer_radius)
        inner_h_over_r = (
            inner_eos.from_surface_density_temperature(
                inner_surface_density,
                inner_temperature,
            ).proper_half_thickness
            / inner_radius
        )
        outer_h_over_r = (
            outer_eos.from_surface_density_temperature(
                outer_surface_density,
                outer_temperature,
            ).proper_half_thickness
            / outer_radius
        )
        target_h_over_r = np.exp(
            (1.0 - fraction) * np.log(inner_h_over_r)
            + fraction * np.log(outer_h_over_r)
        )
        temperature = np.empty_like(radius)
        for index, (local_radius, local_sigma, local_h_over_r) in enumerate(
            zip(radius, sigma, target_h_over_r, strict=True)
        ):
            eos = context.vertical_frequency.eos(float(local_radius))
            lower = float(np.log(eos.minimum_temperature))
            upper = float(np.log(eos.maximum_temperature))

            def h_over_r(log_temperature: float) -> float:
                local_temperature = float(
                    np.clip(
                        np.exp(log_temperature),
                        eos.minimum_temperature,
                        eos.maximum_temperature,
                    )
                )
                return (
                    eos.from_surface_density_temperature(
                        float(local_sigma),
                        local_temperature,
                    ).proper_half_thickness
                    / float(local_radius)
                )

            if (
                h_over_r(lower) >= local_h_over_r
                or h_over_r(upper) <= local_h_over_r
            ):
                raise ValueError(
                    "target seed H/R lies outside the EOS temperature range"
                )
            for _iteration in range(80):
                midpoint = 0.5 * (lower + upper)
                if h_over_r(midpoint) < local_h_over_r:
                    lower = midpoint
                else:
                    upper = midpoint
            temperature[index] = np.exp(0.5 * (lower + upper))
    else:
        temperature = np.exp(
            (1.0 - fraction) * np.log(inner_temperature)
            + fraction * np.log(outer_temperature)
        )
    outer_radius = float(context.grid.edges[-1])
    outer_geometry = kerr_schild_column_geometry(
        outer_radius,
        context.grid.gravitational_radius,
    )
    outer_radial = (
        2.0 * context.grid.gravitational_radius / outer_radius
        + float(outer_radial_velocity_margin_over_c)
    )
    outer_azimuthal = (
        np.sqrt(context.grid.gravitational_radius / outer_radius)
        / outer_geometry.base.lapse
    )
    beta_r = (
        (1.0 - fraction) * inner_radial_velocity_over_c
        + fraction * outer_radial
    )
    beta_phi = (
        (1.0 - fraction) * inner_azimuthal_velocity_over_c
        + fraction * outer_azimuthal
    )
    primitives = np.column_stack(
        (
            np.log(sigma),
            beta_r,
            beta_phi,
            np.log(temperature),
            np.zeros(radius.size),
        )
    )
    for index, local_radius in enumerate(radius):
        _geometry, thermodynamics, primitive = _primitive_from_chart(
            context,
            float(local_radius),
            primitives[index],
        )
        primitives[index, 4] = _closure(
            context,
            float(local_radius),
            thermodynamics,
            primitive,
        ).equilibrium_specific_stress
    return causal_five_field_state_from_primitives(context, primitives)


def causal_five_field_dae_scaling(
    state: CausalFiveFieldDAEState,
    evaluation: CausalFiveFieldDAEEvaluation,
) -> CausalFiveFieldDAEScaling:
    """Return state-aware diagonal scales without changing the equations."""

    state = state.validated()
    n_cells = state.n_cells
    count = causal_five_field_dae_count(n_cells)
    conserved_scale = np.maximum(np.abs(state.conserved), 1.0e-30)
    component_conserved_floor = np.maximum(
        np.median(conserved_scale, axis=0),
        1.0e-30,
    )
    conserved_scale = np.maximum(
        conserved_scale,
        component_conserved_floor[None, :] * 1.0e-6,
    )
    primitive_scale = np.ones_like(state.primitives)
    primitive_scale[:, 4] = np.maximum(
        np.abs(state.primitives[:, 4]),
        max(float(np.median(np.abs(state.primitives[:, 4]))), 1.0e-14),
    )
    face_scale = np.maximum(
        np.abs(state.weighted_face_fluxes_over_c),
        1.0e-30,
    )
    component_face_floor = np.maximum(
        np.median(face_scale, axis=0),
        1.0e-30,
    )
    face_scale = np.maximum(
        face_scale,
        component_face_floor[None, :] * 1.0e-6,
    )
    column_scales = np.concatenate(
        (
            conserved_scale.ravel(),
            primitive_scale.ravel(),
            face_scale.ravel(),
        )
    )

    conservation_scale = np.maximum.reduce(
        (
            np.abs(
                state.weighted_face_fluxes_over_c[1:]
                - state.weighted_face_fluxes_over_c[:-1]
            ),
            np.abs(evaluation.integrated_sources_per_ct),
            np.maximum(
                face_scale[1:],
                face_scale[:-1],
            ),
        )
    )
    conservation_scale = np.maximum(
        conservation_scale,
        np.median(conservation_scale, axis=0)[None, :] * 1.0e-8,
    )
    row_scales = np.concatenate(
        (
            conservation_scale.ravel(),
            conserved_scale.ravel(),
            face_scale[1:-1].ravel(),
            face_scale[0],
            face_scale[-1],
        )
    )
    scaling = CausalFiveFieldDAEScaling(
        column_scales=column_scales,
        row_scales=row_scales,
    )
    return scaling.validated_for(count.total_unknowns)


def audit_causal_five_field_dae_jacobian(
    residual_function,
    vector: np.ndarray,
    scaling: CausalFiveFieldDAEScaling,
    *,
    finite_difference_step: float = 2.0e-6,
    rank_relative_threshold: float = 2.0e-9,
) -> CausalFiveFieldJacobianAudit:
    """Audit a square scaled Jacobian by dense central differences."""

    base = np.asarray(vector, dtype=float)
    scaling = scaling.validated_for(base.size)
    step = float(finite_difference_step)
    if not np.isfinite(step) or not 0.0 < step < 1.0e-2:
        raise ValueError("finite-difference step must be positive and small")
    columns = np.empty((base.size, base.size), dtype=float)
    for index in range(base.size):
        delta = step * scaling.column_scales[index]
        plus = np.array(base, copy=True)
        minus = np.array(base, copy=True)
        plus[index] += delta
        minus[index] -= delta
        columns[:, index] = (
            np.asarray(residual_function(plus), dtype=float)
            - np.asarray(residual_function(minus), dtype=float)
        ) / (2.0 * step * scaling.row_scales)
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        columns,
        full_matrices=False,
    )
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    threshold = max(
        rank_relative_threshold * largest,
        np.finfo(float).eps * base.size * largest,
    )
    rank = int(np.sum(singular_values > threshold))
    return CausalFiveFieldJacobianAudit(
        dimensions=columns.shape,
        numerical_rank=rank,
        singular_values=np.asarray(singular_values, dtype=float),
        smallest_singular_value=smallest,
        largest_singular_value=largest,
        condition_estimate=float(
            largest / max(smallest, np.finfo(float).tiny)
        ),
        finite_difference_step=step,
        scaled_jacobian=np.asarray(columns, dtype=float),
        weakest_right_singular_vector=np.asarray(
            right_vectors[-1],
            dtype=float,
        ),
        weakest_left_singular_vector=np.asarray(
            left_vectors[:, -1],
            dtype=float,
        ),
    )


def _matrix_rank_and_condition(
    values: np.ndarray,
    relative_threshold: float,
) -> tuple[np.ndarray, int, float]:
    singular = np.linalg.svd(values, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    threshold = max(
        relative_threshold * largest,
        np.finfo(float).eps * max(values.shape) * largest,
    )
    rank = int(np.sum(singular > threshold))
    condition = float(
        largest / max(float(singular[-1]), np.finfo(float).tiny)
    )
    return np.asarray(singular, dtype=float), rank, condition


def _outer_thermal_stress_response(
    reduced_jacobian: np.ndarray,
    n_cells: int,
    relative_threshold: float,
) -> CausalFiveFieldOuterThermalStressAudit:
    target = np.asarray(
        [
            _N_FIELDS * (n_cells - 1) + 3,
            _N_FIELDS * (n_cells - 1) + 4,
        ],
        dtype=int,
    )
    all_indices = np.arange(_N_FIELDS * n_cells)
    interior = np.setdiff1d(all_indices, target, assume_unique=True)
    interior_block = reduced_jacobian[np.ix_(interior, interior)]
    coupling_to_target = reduced_jacobian[np.ix_(interior, target)]
    target_from_interior = reduced_jacobian[np.ix_(target, interior)]
    target_block = reduced_jacobian[np.ix_(target, target)]
    (
        _interior_singular,
        interior_rank,
        interior_condition,
    ) = _matrix_rank_and_condition(
        interior_block,
        relative_threshold,
    )
    if interior_rank == interior.size:
        interior_response = np.linalg.solve(
            interior_block,
            coupling_to_target,
        )
    else:
        interior_response = np.linalg.pinv(
            interior_block,
            rcond=relative_threshold,
        ) @ coupling_to_target
    response = target_block - target_from_interior @ interior_response
    singular, rank, condition = _matrix_rank_and_condition(
        response,
        relative_threshold,
    )
    return CausalFiveFieldOuterThermalStressAudit(
        interior_dimensions=interior_block.shape,
        interior_numerical_rank=interior_rank,
        interior_condition_estimate=interior_condition,
        response_matrix=np.asarray(response, dtype=float),
        singular_values=singular,
        numerical_rank=rank,
        condition_estimate=condition,
        determinant=float(np.linalg.det(response)),
    )


def audit_causal_five_field_reduced_stationary_response(
    context: CausalFiveFieldDAEContext,
    state: CausalFiveFieldDAEState,
    full_audit: CausalFiveFieldJacobianAudit,
    *,
    scaling: CausalFiveFieldDAEScaling | None = None,
    finite_difference_step: float = 2.0e-6,
    rank_relative_threshold: float = 1.0e-11,
) -> CausalFiveFieldReducedJacobianAudit:
    """Audit the exact primitive Schur response against direct differences."""

    context = context.validated()
    state = state.validated()
    n_cells = state.n_cells
    if n_cells != int(context.grid.centers.size):
        raise ValueError("state and reduced-audit context use different grids")
    count = causal_five_field_dae_count(n_cells)
    if full_audit.dimensions != (
        count.total_unknowns,
        count.total_unknowns,
    ):
        raise ValueError("full Jacobian audit has incompatible dimensions")
    step = float(finite_difference_step)
    if not np.isfinite(step) or not 0.0 < step < 1.0e-2:
        raise ValueError("finite-difference step must be positive and small")
    threshold = float(rank_relative_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("rank threshold must be positive and finite")

    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    if scaling is None:
        scaling = causal_five_field_dae_scaling(state, evaluation)
    scaling = scaling.validated_for(count.total_unknowns)
    n_reduced = _N_FIELDS * n_cells
    conserved_columns = np.arange(0, n_reduced)
    primitive_columns = np.arange(n_reduced, 2 * n_reduced)
    face_columns = np.arange(2 * n_reduced, count.total_unknowns)
    algebraic_columns = np.concatenate(
        (conserved_columns, face_columns)
    )
    conservation_rows = np.arange(0, n_reduced)
    algebraic_rows = np.arange(n_reduced, count.total_rows)
    full = np.asarray(full_audit.scaled_jacobian, dtype=float)
    algebraic_block = full[np.ix_(algebraic_rows, algebraic_columns)]
    algebraic_from_primitives = full[
        np.ix_(algebraic_rows, primitive_columns)
    ]
    conservation_from_algebraic = full[
        np.ix_(conservation_rows, algebraic_columns)
    ]
    conservation_from_primitives = full[
        np.ix_(conservation_rows, primitive_columns)
    ]
    (
        _algebraic_singular,
        algebraic_rank,
        algebraic_condition,
    ) = _matrix_rank_and_condition(algebraic_block, threshold)
    if algebraic_rank != algebraic_block.shape[0]:
        raise ValueError("primitive/face identity block is not invertible")
    algebraic_response = np.linalg.solve(
        algebraic_block,
        algebraic_from_primitives,
    )
    schur = (
        conservation_from_primitives
        - conservation_from_algebraic @ algebraic_response
    )

    primitive_scale = scaling.column_scales[primitive_columns]
    conservation_scale = scaling.row_scales[conservation_rows]
    base_primitives = np.asarray(state.primitives, dtype=float).ravel()
    direct = np.empty((n_reduced, n_reduced), dtype=float)
    baseline_choked = bool(evaluation.outer_boundary_choked)
    for index in range(n_reduced):
        delta = step * primitive_scale[index]
        plus = np.array(base_primitives, copy=True)
        minus = np.array(base_primitives, copy=True)
        plus[index] += delta
        minus[index] -= delta
        plus_state = causal_five_field_state_from_primitives(
            context,
            plus.reshape(n_cells, _N_FIELDS),
        )
        minus_state = causal_five_field_state_from_primitives(
            context,
            minus.reshape(n_cells, _N_FIELDS),
        )
        plus_evaluation = evaluate_causal_five_field_dae(
            pack_causal_five_field_state(plus_state),
            context,
        )
        minus_evaluation = evaluate_causal_five_field_dae(
            pack_causal_five_field_state(minus_state),
            context,
        )
        if (
            bool(plus_evaluation.outer_boundary_choked) != baseline_choked
            or bool(minus_evaluation.outer_boundary_choked)
            != baseline_choked
        ):
            raise ValueError(
                "reduced finite difference crossed the Roche active set"
            )
        direct[:, index] = (
            plus_evaluation.conservation_rows.ravel()
            - minus_evaluation.conservation_rows.ravel()
        ) / (2.0 * step * conservation_scale)

    left, singular, right = np.linalg.svd(direct, full_matrices=False)
    largest = float(singular[0])
    smallest = float(singular[-1])
    rank_threshold = max(
        threshold * largest,
        np.finfo(float).eps * n_reduced * largest,
    )
    rank = int(np.sum(singular > rank_threshold))
    difference = direct - schur
    maximum_absolute_defect = float(np.max(np.abs(difference)))
    relative_frobenius_defect = float(
        np.linalg.norm(difference)
        / max(
            np.linalg.norm(direct),
            np.linalg.norm(schur),
            np.finfo(float).tiny,
        )
    )
    directions = [
        np.ones(n_reduced, dtype=float),
        np.linspace(-1.0, 1.0, n_reduced),
        np.eye(1, n_reduced, n_reduced - 2, dtype=float).ravel(),
        np.eye(1, n_reduced, n_reduced - 1, dtype=float).ravel(),
        np.asarray(right[-1], dtype=float),
    ]
    directional_defects = []
    operator_scaled_directional_defects = []
    for direction in directions:
        normalized = direction / max(
            np.linalg.norm(direction),
            np.finfo(float).tiny,
        )
        direct_product = direct @ normalized
        schur_product = schur @ normalized
        directional_defects.append(
            np.linalg.norm(direct_product - schur_product)
            / max(
                np.linalg.norm(direct_product),
                np.linalg.norm(schur_product),
                np.finfo(float).tiny,
            )
        )
        operator_scaled_directional_defects.append(
            np.linalg.norm(direct_product - schur_product)
            / max(largest, np.finfo(float).tiny)
        )

    reduced_right = np.asarray(right[-1], dtype=float)
    algebraic_direction = -np.linalg.solve(
        algebraic_block,
        algebraic_from_primitives @ reduced_right,
    )
    reconstructed = np.zeros(count.total_unknowns, dtype=float)
    reconstructed[primitive_columns] = reduced_right
    reconstructed[algebraic_columns] = algebraic_direction
    reconstructed /= max(
        np.linalg.norm(reconstructed),
        np.finfo(float).tiny,
    )
    algebraic_residual_norm = float(
        np.linalg.norm(full[algebraic_rows] @ reconstructed)
    )
    full_residual_norm = float(np.linalg.norm(full @ reconstructed))
    alignment = float(
        abs(
            np.dot(
                reconstructed,
                np.asarray(
                    full_audit.weakest_right_singular_vector,
                    dtype=float,
                ),
            )
        )
    )
    outer_response = _outer_thermal_stress_response(
        direct,
        n_cells,
        threshold,
    )
    schur_singular, schur_rank, schur_condition = (
        _matrix_rank_and_condition(schur, threshold)
    )
    return CausalFiveFieldReducedJacobianAudit(
        dimensions=direct.shape,
        numerical_rank=rank,
        singular_values=np.asarray(singular, dtype=float),
        smallest_singular_value=smallest,
        largest_singular_value=largest,
        condition_estimate=float(
            largest / max(smallest, np.finfo(float).tiny)
        ),
        finite_difference_step=step,
        direct_scaled_jacobian=np.asarray(direct, dtype=float),
        schur_scaled_jacobian=np.asarray(schur, dtype=float),
        schur_singular_values=schur_singular,
        schur_numerical_rank=schur_rank,
        schur_condition_estimate=schur_condition,
        maximum_absolute_matrix_defect=maximum_absolute_defect,
        relative_frobenius_matrix_defect=relative_frobenius_defect,
        maximum_directional_relative_defect=float(
            max(directional_defects)
        ),
        maximum_directional_operator_scaled_defect=float(
            max(operator_scaled_directional_defects)
        ),
        algebraic_dimensions=algebraic_block.shape,
        algebraic_numerical_rank=algebraic_rank,
        algebraic_condition_estimate=algebraic_condition,
        reconstructed_algebraic_residual_norm=algebraic_residual_norm,
        reconstructed_full_residual_norm=full_residual_norm,
        full_weakest_vector_alignment=alignment,
        weakest_right_singular_vector=reduced_right,
        weakest_left_singular_vector=np.asarray(left[:, -1], dtype=float),
        reconstructed_full_scaled_vector=reconstructed,
        outer_thermal_stress=outer_response,
        outer_boundary_choked=baseline_choked,
    )


def audit_causal_five_field_consistent_initial_data(
    context: CausalFiveFieldDAEContext,
    state: CausalFiveFieldDAEState,
    stationary_audit: CausalFiveFieldJacobianAudit,
    backward_euler_audit: CausalFiveFieldJacobianAudit,
    *,
    scaling: CausalFiveFieldDAEScaling | None = None,
    descriptor_timestep_seconds: float = 1.0,
    rank_relative_threshold: float = 1.0e-11,
) -> CausalFiveFieldConsistentInitialDataAudit:
    """Solve the initial storage balance on the algebraic tangent manifold."""

    context = context.validated()
    state = state.validated()
    n_cells = state.n_cells
    count = causal_five_field_dae_count(n_cells)
    expected_dimensions = (count.total_unknowns, count.total_unknowns)
    if (
        stationary_audit.dimensions != expected_dimensions
        or backward_euler_audit.dimensions != expected_dimensions
    ):
        raise ValueError("consistent-data audits have incompatible dimensions")
    timestep = float(descriptor_timestep_seconds)
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("descriptor timestep must be positive and finite")
    threshold = float(rank_relative_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("rank threshold must be positive and finite")

    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    if scaling is None:
        scaling = causal_five_field_dae_scaling(state, evaluation)
    scaling = scaling.validated_for(count.total_unknowns)
    n_differential = _N_FIELDS * n_cells
    conservation_rows = np.arange(0, n_differential)
    algebraic_rows = np.arange(n_differential, count.total_rows)
    descriptor = timestep * (
        np.asarray(backward_euler_audit.scaled_jacobian, dtype=float)
        - np.asarray(stationary_audit.scaled_jacobian, dtype=float)
    )
    descriptor_rows = descriptor[conservation_rows]
    (
        _descriptor_singular,
        descriptor_rank,
        _descriptor_condition,
    ) = _matrix_rank_and_condition(descriptor_rows, threshold)
    algebraic_tangent = np.asarray(
        stationary_audit.scaled_jacobian[algebraic_rows],
        dtype=float,
    )
    consistency_matrix = np.vstack(
        (descriptor_rows, algebraic_tangent)
    )
    singular, rank, condition = _matrix_rank_and_condition(
        consistency_matrix,
        threshold,
    )
    if rank != count.total_unknowns:
        raise ValueError("consistent-initial-data matrix is not invertible")
    scaled_residual = np.asarray(
        evaluation.residual / scaling.row_scales,
        dtype=float,
    )
    right_hand_side = np.concatenate(
        (
            -scaled_residual[conservation_rows],
            np.zeros(algebraic_rows.size, dtype=float),
        )
    )
    tangent = np.linalg.solve(consistency_matrix, right_hand_side)
    consistency_residual = consistency_matrix @ tangent - right_hand_side
    storage_residual = (
        descriptor_rows @ tangent
        + scaled_residual[conservation_rows]
    )
    algebraic_residual = algebraic_tangent @ tangent
    primitive_columns = slice(n_differential, 2 * n_differential)
    return CausalFiveFieldConsistentInitialDataAudit(
        dimensions=consistency_matrix.shape,
        numerical_rank=rank,
        singular_values=singular,
        condition_estimate=condition,
        descriptor_dimensions=descriptor_rows.shape,
        descriptor_numerical_rank=descriptor_rank,
        maximum_initial_algebraic_residual=float(
            np.max(np.abs(scaled_residual[algebraic_rows]))
        ),
        maximum_scaled_consistency_residual=float(
            np.max(np.abs(consistency_residual))
        ),
        scaled_tangent=np.asarray(tangent, dtype=float),
        maximum_scaled_tangent=float(np.max(np.abs(tangent))),
        maximum_scaled_primitive_tangent=float(
            np.max(np.abs(tangent[primitive_columns]))
        ),
        storage_balance_residual_norm=float(np.linalg.norm(storage_residual)),
        algebraic_tangent_residual_norm=float(
            np.linalg.norm(algebraic_residual)
        ),
    )
