"""Assembled five-field causal Kerr-Schild finite-volume DAE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C, DEFAULT_KAPPA_ES

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
    integrated_sources_per_ct: np.ndarray
    proper_shear_rates: np.ndarray
    proper_log_height_rates: np.ndarray
    scattering_optical_depths: np.ndarray
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


def _interior_rusanov_flux(
    context: CausalFiveFieldDAEContext,
    face_index: int,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
) -> np.ndarray:
    """Return one proper-measure weighted five-field Rusanov flux."""

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
    flux = (
        0.5 * (left.flux_over_c + right.flux_over_c)
        - 0.5 * maximum_speed * (right.conserved - left.conserved)
    )
    return np.asarray(
        context.grid.face_measures[face_index] * flux,
        dtype=float,
    )


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
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-integrated five-field sources per coordinate ``ct``."""

    n_cells = len(cell_states)
    sources = np.zeros((n_cells, _N_FIELDS), dtype=float)
    optical_depths = np.full(n_cells, np.nan, dtype=float)
    for index, state in enumerate(cell_states):
        perfect = audit_kerr_schild_column_sources(
            state.geometry,
            state.primitive,
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
            optical_depths[index] = thermal.scattering_optical_depth
        else:
            vertical_work = (
                -state.thermodynamics.integrated_pressure
                * height_rates[index]
            )
            local += causal_comoving_energy_source(
                state.geometry,
                state.primitive,
                comoving_energy_rate=float(vertical_work),
            ).killing_source_per_ct
            optical_depths[index] = (
                0.5
                * context.kappa
                * state.thermodynamics.surface_density
            )

        sources[index, :4] = (
            context.grid.cell_measures[index] * local
        )
        sources[index, 4] = (
            context.grid.cell_measures[index]
            * causal_stress_relaxation_source(
                state.geometry,
                state.stress,
                state.closure,
                positive_shear_rate=float(shear_rates[index]),
            )
        )
    if context.stream_sources is not None:
        sources[:, :4] += (
            context.stream_sources.weighted_killing_source_per_ct
        )
    return sources, optical_depths


def _mapped_state_and_fluxes(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
) -> tuple[
    list[CausalFiveFieldCellState],
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
    faces[0] = _inner_face_flux(context, primitive_charts[0])
    for face in range(1, n_cells):
        faces[face] = _interior_rusanov_flux(
            context,
            face,
            primitive_charts[face - 1],
            primitive_charts[face],
        )
    faces[-1], choked, incoming = _outer_face_flux(
        context,
        primitive_charts[-1],
    )
    return cell_states, mapped, faces, choked, incoming


def evaluate_causal_five_field_dae(
    vector: np.ndarray,
    context: CausalFiveFieldDAEContext,
    *,
    old_vector: np.ndarray | None = None,
    timestep_seconds: float | None = None,
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

    cell_states, mapped, numerical_fluxes, choked, incoming = (
        _mapped_state_and_fluxes(context, state.primitives)
    )
    shear_rates, height_rates = _straight_path_cell_rates(
        context,
        cell_states,
    )
    sources, optical_depths = _integrated_cell_sources(
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
    temporal_storage = np.zeros((n_cells, 4), dtype=float)
    if old_vector is not None:
        assert timestep_seconds is not None
        old = unpack_causal_five_field_state(old_vector, n_cells)
        coordinate_timestep = C * timestep_seconds
        conservation += (
            context.grid.cell_measures[:, None]
            * (state.conserved - old.conserved)
            / coordinate_timestep
        )
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
            temporal_storage[index] = (
                context.grid.cell_measures[index]
                * storage.killing_storage_increment
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
        integrated_sources_per_ct=sources,
        proper_shear_rates=shear_rates,
        proper_log_height_rates=height_rates,
        scattering_optical_depths=optical_depths,
        temporal_vertical_storage=temporal_storage,
        outer_boundary_choked=choked,
        outer_incoming_characteristics=incoming,
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
    _states, mapped, faces, _choked, _incoming = (
        _mapped_state_and_fluxes(context, primitives)
    )
    return CausalFiveFieldDAEState(
        conserved=mapped,
        primitives=np.array(primitives, copy=True),
        weighted_face_fluxes_over_c=faces,
    ).validated()


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
) -> CausalFiveFieldDAEState:
    """Return a smooth low-throughput, alpha-equilibrium preflight seed."""

    context = context.validated()
    radius = np.asarray(context.grid.centers, dtype=float)
    fraction = (
        np.log(radius / radius[0]) / np.log(radius[-1] / radius[0])
        if radius.size > 1
        else np.zeros(1)
    )
    sigma = np.exp(
        (1.0 - fraction) * np.log(inner_surface_density)
        + fraction * np.log(outer_surface_density)
    )
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
