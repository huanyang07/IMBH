"""Candidate internal moment coordinates for the WP10c8i audit.

The coordinate rows in this module act on the *scaled primitive tangent*
returned by :func:`causal_five_field_reduced_descriptor_matrices`.  Conserved
and face-flux rows are reconstructed through that descriptor's exact
algebraic Schur response.  Consequently the shell energy coordinates below
are instantaneous Killing-energy coordinates.  Cumulative responsive-height
work is deliberately absent: it remains a vector storage one-form and a path
ledger, not an instantaneous state function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C

from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    unpack_causal_five_field_state,
)


CAUSAL_WP10C8I_DEFAULT_SHAPE_BANDS_RG = (
    ("6_to_60rg", 6.0, 60.0),
    ("source_shell", 200.0, 280.0),
)
CAUSAL_WP10C8I_STORAGE_SEMANTICS = (
    "instantaneous_conserved_storage_without_cumulative_vertical_work"
)

_INSTANTANEOUS_SHELL_COMPONENTS = (
    (0, "rest_mass"),
    (2, "angular_momentum"),
    (3, "killing_energy"),
)
_INTERFACE_FLUX_COMPONENTS = _INSTANTANEOUS_SHELL_COMPONENTS
_RADIAL_MOMENTUM_COMPONENT = (1, "radial_momentum")
_STRESS_STORAGE_COMPONENT = (4, "stress_storage")
_FIELD_COUNT = 5


@dataclass(frozen=True)
class CausalMomentShellGeometry:
    """Mesh-coincident shell geometry used by one candidate ladder."""

    edges_rg: np.ndarray
    edge_indices: np.ndarray
    cell_masks: tuple[np.ndarray, ...]
    shell_names: tuple[str, ...]

    @property
    def shell_count(self) -> int:
        return len(self.cell_masks)


@dataclass(frozen=True)
class CausalMomentCoordinateLevel:
    """One cumulative candidate coordinate set.

    ``raw_constraint_matrix`` contains physical coordinate derivatives with
    respect to the scaled primitive tangent.  ``constraint_matrix`` divides
    every row by its declared natural coordinate scale.  The final
    ``conditioned_constraint_matrix`` additionally normalizes each nonzero
    row to unit Euclidean norm; it is suitable for rank diagnostics while
    preserving exactly the same null space.
    """

    name: str
    coordinate_names: tuple[str, ...]
    coordinate_families: tuple[str, ...]
    coordinate_values: np.ndarray
    coordinate_scales: np.ndarray
    raw_constraint_matrix: np.ndarray
    constraint_matrix: np.ndarray
    constraint_row_norms: np.ndarray
    conditioned_constraint_matrix: np.ndarray

    @property
    def coordinate_count(self) -> int:
        return len(self.coordinate_names)


@dataclass(frozen=True)
class CausalMomentCoordinateValueLevel:
    """One cumulative candidate coordinate set without tangent rows."""

    name: str
    coordinate_names: tuple[str, ...]
    coordinate_families: tuple[str, ...]
    coordinate_values: np.ndarray
    coordinate_scales: np.ndarray

    @property
    def coordinate_count(self) -> int:
        return len(self.coordinate_names)


@dataclass(frozen=True)
class CausalFiveFieldMomentCoordinateValues:
    """Value-only WP10c8i coordinates and macro-interface fluxes.

    Unlike :class:`CausalFiveFieldMomentCoordinateLadder`, this object does
    not require a reduced descriptor.  It is therefore suitable for exact
    nonlinear equal-coordinate lifting, where only the finite-state
    coordinate map and interface observables are needed.
    """

    geometry: CausalMomentShellGeometry
    levels: tuple[CausalMomentCoordinateValueLevel, ...]
    storage_semantics: str
    interface_flux_names: tuple[str, ...]
    interface_flux_values: np.ndarray
    interface_flux_scales: np.ndarray

    def level(self, name: str) -> CausalMomentCoordinateValueLevel:
        """Return a named cumulative value-only level."""

        for level in self.levels:
            if level.name == name:
                return level
        raise KeyError(name)


@dataclass(frozen=True)
class CausalFiveFieldMomentCoordinateLadder:
    """Incremental WP10c8i candidate coordinates and interface outputs."""

    geometry: CausalMomentShellGeometry
    levels: tuple[CausalMomentCoordinateLevel, ...]
    storage_semantics: str
    primitive_column_scales: np.ndarray
    interface_flux_names: tuple[str, ...]
    interface_flux_values: np.ndarray
    interface_flux_scales: np.ndarray
    raw_interface_flux_jacobian: np.ndarray
    interface_flux_jacobian: np.ndarray

    def level(self, name: str) -> CausalMomentCoordinateLevel:
        """Return a named cumulative level."""

        for level in self.levels:
            if level.name == name:
                return level
        raise KeyError(name)


def _finite_vector(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        (shape is not None and array.shape != shape)
        or np.any(~np.isfinite(array))
    ):
        raise ValueError(f"{name} has an invalid shape or value")
    return array


def _coordinate_scale(
    value: float,
    contributions: np.ndarray,
) -> float:
    """Return a cancellation-safe physical scale for one integral."""

    l1_scale = float(np.sum(np.abs(np.asarray(contributions, dtype=float))))
    scale = max(abs(float(value)), l1_scale, np.finfo(float).tiny)
    if not np.isfinite(scale):
        raise ValueError("coordinate scale is not finite")
    return scale


def causal_mesh_coincident_moment_shells(
    context: CausalFiveFieldDAEContext,
    shell_edges_rg: np.ndarray,
    *,
    edge_tolerance_rg: float = 1.0e-10,
) -> CausalMomentShellGeometry:
    """Validate and return a complete mesh-coincident radial shell layout."""

    context = context.validated()
    grid_edges_rg = (
        np.asarray(context.grid.edges, dtype=float)
        / context.grid.gravitational_radius
    )
    declared = np.asarray(shell_edges_rg, dtype=float)
    tolerance = float(edge_tolerance_rg)
    if (
        declared.ndim != 1
        or declared.size < 2
        or np.any(~np.isfinite(declared))
        or np.any(np.diff(declared) <= 0.0)
        or not np.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("shell edges and tolerance are invalid")
    if not (
        np.isclose(
            declared[0],
            grid_edges_rg[0],
            rtol=0.0,
            atol=tolerance,
        )
        and np.isclose(
            declared[-1],
            grid_edges_rg[-1],
            rtol=0.0,
            atol=tolerance,
        )
    ):
        raise ValueError("shell layout must cover the complete radial mesh")

    indices = []
    snapped = []
    for edge in declared:
        distances = np.abs(grid_edges_rg - edge)
        index = int(np.argmin(distances))
        if distances[index] > tolerance:
            raise ValueError("a declared shell edge is not mesh coincident")
        indices.append(index)
        snapped.append(float(grid_edges_rg[index]))
    edge_indices = np.asarray(indices, dtype=int)
    if (
        len(set(indices)) != len(indices)
        or np.any(np.diff(edge_indices) <= 0)
    ):
        raise ValueError("shell edges do not define nonempty mesh intervals")

    n_cells = int(context.grid.centers.size)
    masks = []
    names = []
    for shell_index, (left, right) in enumerate(
        zip(edge_indices[:-1], edge_indices[1:], strict=True)
    ):
        mask = np.zeros(n_cells, dtype=bool)
        mask[left:right] = True
        if not np.any(mask):
            raise ValueError("a declared moment shell is empty")
        masks.append(mask)
        names.append(f"shell_{shell_index}")
    return CausalMomentShellGeometry(
        edges_rg=np.asarray(snapped, dtype=float),
        edge_indices=edge_indices,
        cell_masks=tuple(masks),
        shell_names=tuple(names),
    )


def _schur_physical_responses(
    reduced_descriptor: dict,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_reduced = _FIELD_COUNT * n_cells
    primitive_scales = _finite_vector(
        reduced_descriptor["primitive_column_scales"],
        name="primitive_column_scales",
        shape=(n_reduced,),
    )
    if np.any(primitive_scales <= 0.0):
        raise ValueError("primitive column scales must be positive")
    algebraic_count = (
        _FIELD_COUNT * n_cells
        + _FIELD_COUNT * (n_cells + 1)
    )
    algebraic_scales = _finite_vector(
        reduced_descriptor["algebraic_column_scales"],
        name="algebraic_column_scales",
        shape=(algebraic_count,),
    )
    if np.any(algebraic_scales <= 0.0):
        raise ValueError("algebraic column scales must be positive")
    response_scaled = _finite_vector(
        reduced_descriptor["algebraic_response_scaled"],
        name="algebraic_response_scaled",
        shape=(algebraic_count, n_reduced),
    )
    response_physical = algebraic_scales[:, None] * response_scaled
    conserved_response = response_physical[:n_reduced].reshape(
        n_cells,
        _FIELD_COUNT,
        n_reduced,
    )
    face_response = response_physical[n_reduced:].reshape(
        n_cells + 1,
        _FIELD_COUNT,
        n_reduced,
    )
    return primitive_scales, conserved_response, face_response


def _direct_primitive_row(
    primitive_scales: np.ndarray,
    n_cells: int,
    component: int,
    coefficients: np.ndarray,
) -> np.ndarray:
    weights = _finite_vector(
        coefficients,
        name="primitive moment coefficients",
        shape=(n_cells,),
    )
    row = np.zeros(_FIELD_COUNT * n_cells, dtype=float)
    component_columns = np.arange(component, row.size, _FIELD_COUNT)
    row[component_columns] = (
        weights * primitive_scales[component_columns]
    )
    return row


def _conditioned_level(
    *,
    name: str,
    coordinate_names: list[str],
    coordinate_families: list[str],
    coordinate_values: list[float],
    coordinate_scales: list[float],
    raw_rows: list[np.ndarray],
) -> CausalMomentCoordinateLevel:
    raw = np.asarray(raw_rows, dtype=float)
    values = np.asarray(coordinate_values, dtype=float)
    scales = np.asarray(coordinate_scales, dtype=float)
    if (
        raw.ndim != 2
        or raw.shape[0] != len(coordinate_names)
        or values.shape != (raw.shape[0],)
        or scales.shape != (raw.shape[0],)
        or len(coordinate_families) != raw.shape[0]
        or np.any(~np.isfinite(raw))
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError("candidate moment rows are inconsistent")
    scaled = raw / scales[:, None]
    row_norms = np.linalg.norm(scaled, axis=1)
    conditioned = np.zeros_like(scaled)
    active = row_norms > 0.0
    conditioned[active] = scaled[active] / row_norms[active, None]
    return CausalMomentCoordinateLevel(
        name=name,
        coordinate_names=tuple(coordinate_names),
        coordinate_families=tuple(coordinate_families),
        coordinate_values=values,
        coordinate_scales=scales,
        raw_constraint_matrix=raw,
        constraint_matrix=scaled,
        constraint_row_norms=row_norms,
        conditioned_constraint_matrix=conditioned,
    )


def _find_shape_shell(
    geometry: CausalMomentShellGeometry,
    lower_rg: float,
    upper_rg: float,
    *,
    absolute_tolerance: float,
    nominal_relative_tolerance: float,
) -> int:
    exact_matches = np.flatnonzero(
        np.isclose(
            geometry.edges_rg[:-1],
            lower_rg,
            rtol=0.0,
            atol=absolute_tolerance,
        )
        & np.isclose(
            geometry.edges_rg[1:],
            upper_rg,
            rtol=0.0,
            atol=absolute_tolerance,
        )
    )
    if exact_matches.size == 1:
        return int(exact_matches[0])

    lower_edge = int(np.argmin(np.abs(geometry.edges_rg - lower_rg)))
    upper_edge = int(np.argmin(np.abs(geometry.edges_rg - upper_rg)))
    relative_errors = np.asarray(
        (
            abs(float(geometry.edges_rg[lower_edge]) - lower_rg)
            / max(abs(lower_rg), np.finfo(float).tiny),
            abs(float(geometry.edges_rg[upper_edge]) - upper_rg)
            / max(abs(upper_rg), np.finfo(float).tiny),
        )
    )
    if (
        upper_edge != lower_edge + 1
        or np.any(relative_errors > nominal_relative_tolerance)
    ):
        raise ValueError(
            "each targeted shape band must resolve to one declared shell"
        )
    return lower_edge


def _value_only_level(
    *,
    name: str,
    coordinate_names: list[str],
    coordinate_families: list[str],
    coordinate_values: list[float],
    coordinate_scales: list[float],
) -> CausalMomentCoordinateValueLevel:
    values = np.asarray(coordinate_values, dtype=float)
    scales = np.asarray(coordinate_scales, dtype=float)
    count = len(coordinate_names)
    if (
        len(coordinate_families) != count
        or values.shape != (count,)
        or scales.shape != (count,)
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError("candidate moment values are inconsistent")
    return CausalMomentCoordinateValueLevel(
        name=name,
        coordinate_names=tuple(coordinate_names),
        coordinate_families=tuple(coordinate_families),
        coordinate_values=values,
        coordinate_scales=scales,
    )


def causal_five_field_moment_coordinate_values(
    context: CausalFiveFieldDAEContext,
    state_vector: np.ndarray,
    shell_edges_rg: np.ndarray,
    *,
    shape_bands_rg: tuple[
        tuple[str, float, float], ...
    ] = CAUSAL_WP10C8I_DEFAULT_SHAPE_BANDS_RG,
    edge_tolerance_rg: float = 1.0e-10,
    nominal_shape_edge_relative_tolerance: float = 0.1,
) -> CausalFiveFieldMomentCoordinateValues:
    """Evaluate the exact finite-state WP10c8i coordinate ladder.

    This value-only evaluator preserves the cumulative 15/20/25/30/34
    ordering and natural scales used by
    :func:`causal_five_field_moment_coordinate_ladder`, but deliberately
    performs no tangent or Schur-response construction.
    """

    context = context.validated()
    nominal_tolerance = float(nominal_shape_edge_relative_tolerance)
    if not np.isfinite(nominal_tolerance) or nominal_tolerance < 0.0:
        raise ValueError(
            "nominal shape-edge relative tolerance must be nonnegative"
        )
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(state_vector, n_cells)
    geometry = causal_mesh_coincident_moment_shells(
        context,
        shell_edges_rg,
        edge_tolerance_rg=edge_tolerance_rg,
    )
    measures = _finite_vector(
        context.grid.cell_measures,
        name="cell measures",
        shape=(n_cells,),
    )
    if np.any(measures <= 0.0):
        raise ValueError("cell measures must be positive")
    radius_rg = _finite_vector(
        context.grid.centers / context.grid.gravitational_radius,
        name="cell radii",
        shape=(n_cells,),
    )
    log_radius = np.log(radius_rg)

    names: list[str] = []
    families: list[str] = []
    values: list[float] = []
    scales: list[float] = []
    levels = []

    def add_conserved_shell_family(
        components: tuple[tuple[int, str], ...],
        family_name: str,
    ) -> None:
        for shell_index, mask in enumerate(geometry.cell_masks):
            for component, component_name in components:
                contributions = (
                    measures[mask] * state.conserved[mask, component]
                )
                value = float(np.sum(contributions))
                names.append(f"shell_{shell_index}_{component_name}")
                families.append(family_name)
                values.append(value)
                scales.append(_coordinate_scale(value, contributions))

    def append_level(name: str) -> None:
        levels.append(
            _value_only_level(
                name=name,
                coordinate_names=names,
                coordinate_families=families,
                coordinate_values=values,
                coordinate_scales=scales,
            )
        )

    add_conserved_shell_family(
        _INSTANTANEOUS_SHELL_COMPONENTS,
        "instantaneous_shell_mje",
    )
    append_level("instantaneous_shell_mje")

    for shell_index, mask in enumerate(geometry.cell_masks):
        shell_weights = measures[mask] / float(np.sum(measures[mask]))
        names.append(f"shell_{shell_index}_mean_log_temperature")
        families.append("shell_mean_log_temperature")
        values.append(float(shell_weights @ state.primitives[mask, 3]))
        scales.append(1.0)
    append_level("plus_shell_mean_log_temperature")

    add_conserved_shell_family(
        (_RADIAL_MOMENTUM_COMPONENT,),
        "shell_radial_momentum",
    )
    append_level("plus_shell_radial_momentum")

    add_conserved_shell_family(
        (_STRESS_STORAGE_COMPONENT,),
        "shell_stress_storage",
    )
    append_level("plus_shell_stress_storage")

    seen_shape_names = set()
    for band_name, lower_rg, upper_rg in shape_bands_rg:
        label = str(band_name)
        lower = float(lower_rg)
        upper = float(upper_rg)
        if (
            not label
            or label in seen_shape_names
            or not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower <= 0.0
            or upper <= lower
        ):
            raise ValueError("targeted shape band is invalid")
        seen_shape_names.add(label)
        shell_index = _find_shape_shell(
            geometry,
            lower,
            upper,
            absolute_tolerance=edge_tolerance_rg,
            nominal_relative_tolerance=nominal_tolerance,
        )
        mask = geometry.cell_masks[shell_index]
        shell_weights = measures[mask] / float(np.sum(measures[mask]))
        centered = log_radius[mask] - float(
            shell_weights @ log_radius[mask]
        )
        radial_rms = float(
            np.sqrt(shell_weights @ np.square(centered))
        )
        if not np.isfinite(radial_rms) or radial_rms <= 0.0:
            raise ValueError(
                "targeted shape shell needs multiple distinct cell radii"
            )
        signed_weights = shell_weights * centered / radial_rms
        for component, field_name in (
            (3, "log_temperature"),
            (0, "log_surface_density"),
        ):
            names.append(f"shape_{label}_{field_name}_first")
            families.append("targeted_first_shape_moment")
            values.append(
                float(signed_weights @ state.primitives[mask, component])
            )
            scales.append(1.0)
    append_level("plus_targeted_shape_moments")

    physical_faces = C * state.weighted_face_fluxes_over_c
    component_flux_scales = {
        component: max(
            float(np.max(np.abs(physical_faces[:, component]))),
            np.finfo(float).tiny,
        )
        for component, _name in _INTERFACE_FLUX_COMPONENTS
    }
    interface_names = []
    interface_values = []
    interface_scales = []
    for boundary_index, face in enumerate(
        geometry.edge_indices[1:-1],
        start=1,
    ):
        for component, component_name in _INTERFACE_FLUX_COMPONENTS:
            interface_names.append(
                f"interface_{boundary_index}_{component_name}"
            )
            interface_values.append(float(physical_faces[face, component]))
            interface_scales.append(component_flux_scales[component])

    return CausalFiveFieldMomentCoordinateValues(
        geometry=geometry,
        levels=tuple(levels),
        storage_semantics=CAUSAL_WP10C8I_STORAGE_SEMANTICS,
        interface_flux_names=tuple(interface_names),
        interface_flux_values=np.asarray(interface_values, dtype=float),
        interface_flux_scales=np.asarray(interface_scales, dtype=float),
    )


def causal_five_field_moment_coordinate_ladder(
    context: CausalFiveFieldDAEContext,
    state_vector: np.ndarray,
    reduced_descriptor: dict,
    shell_edges_rg: np.ndarray,
    *,
    shape_bands_rg: tuple[
        tuple[str, float, float], ...
    ] = CAUSAL_WP10C8I_DEFAULT_SHAPE_BANDS_RG,
    edge_tolerance_rg: float = 1.0e-10,
    nominal_shape_edge_relative_tolerance: float = 0.1,
) -> CausalFiveFieldMomentCoordinateLadder:
    """Build the incremental, instantaneous WP10c8i coordinate ladder.

    The returned levels are cumulative and ordered as:

    1. shell rest mass, angular momentum, and Killing energy;
    2. shell mean log temperature;
    3. shell radial momentum;
    4. shell causal-stress storage ``Q_chi``;
    5. centered first log-temperature and log-density shape moments in the
       declared target shells.
    """

    context = context.validated()
    nominal_tolerance = float(nominal_shape_edge_relative_tolerance)
    if not np.isfinite(nominal_tolerance) or nominal_tolerance < 0.0:
        raise ValueError(
            "nominal shape-edge relative tolerance must be nonnegative"
        )
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(state_vector, n_cells)
    geometry = causal_mesh_coincident_moment_shells(
        context,
        shell_edges_rg,
        edge_tolerance_rg=edge_tolerance_rg,
    )
    (
        primitive_scales,
        conserved_response,
        face_response,
    ) = _schur_physical_responses(reduced_descriptor, n_cells)
    measures = _finite_vector(
        context.grid.cell_measures,
        name="cell measures",
        shape=(n_cells,),
    )
    if np.any(measures <= 0.0):
        raise ValueError("cell measures must be positive")
    radius_rg = _finite_vector(
        context.grid.centers / context.grid.gravitational_radius,
        name="cell radii",
        shape=(n_cells,),
    )
    log_radius = np.log(radius_rg)

    names: list[str] = []
    families: list[str] = []
    values: list[float] = []
    scales: list[float] = []
    rows: list[np.ndarray] = []
    levels = []

    def add_conserved_shell_family(
        components: tuple[tuple[int, str], ...],
        family_name: str,
    ) -> None:
        for shell_index, mask in enumerate(geometry.cell_masks):
            for component, component_name in components:
                contributions = (
                    measures[mask] * state.conserved[mask, component]
                )
                value = float(np.sum(contributions))
                row = np.sum(
                    measures[mask, None]
                    * conserved_response[mask, component, :],
                    axis=0,
                )
                names.append(f"shell_{shell_index}_{component_name}")
                families.append(family_name)
                values.append(value)
                scales.append(_coordinate_scale(value, contributions))
                rows.append(row)

    add_conserved_shell_family(
        _INSTANTANEOUS_SHELL_COMPONENTS,
        "instantaneous_shell_mje",
    )
    levels.append(
        _conditioned_level(
            name="instantaneous_shell_mje",
            coordinate_names=names,
            coordinate_families=families,
            coordinate_values=values,
            coordinate_scales=scales,
            raw_rows=rows,
        )
    )

    for shell_index, mask in enumerate(geometry.cell_masks):
        shell_weights = measures[mask] / float(np.sum(measures[mask]))
        coefficients = np.zeros(n_cells, dtype=float)
        coefficients[mask] = shell_weights
        names.append(f"shell_{shell_index}_mean_log_temperature")
        families.append("shell_mean_log_temperature")
        values.append(
            float(shell_weights @ state.primitives[mask, 3])
        )
        scales.append(1.0)
        rows.append(
            _direct_primitive_row(
                primitive_scales,
                n_cells,
                3,
                coefficients,
            )
        )
    levels.append(
        _conditioned_level(
            name="plus_shell_mean_log_temperature",
            coordinate_names=names,
            coordinate_families=families,
            coordinate_values=values,
            coordinate_scales=scales,
            raw_rows=rows,
        )
    )

    add_conserved_shell_family(
        (_RADIAL_MOMENTUM_COMPONENT,),
        "shell_radial_momentum",
    )
    levels.append(
        _conditioned_level(
            name="plus_shell_radial_momentum",
            coordinate_names=names,
            coordinate_families=families,
            coordinate_values=values,
            coordinate_scales=scales,
            raw_rows=rows,
        )
    )

    add_conserved_shell_family(
        (_STRESS_STORAGE_COMPONENT,),
        "shell_stress_storage",
    )
    levels.append(
        _conditioned_level(
            name="plus_shell_stress_storage",
            coordinate_names=names,
            coordinate_families=families,
            coordinate_values=values,
            coordinate_scales=scales,
            raw_rows=rows,
        )
    )

    seen_shape_names = set()
    for band_name, lower_rg, upper_rg in shape_bands_rg:
        label = str(band_name)
        lower = float(lower_rg)
        upper = float(upper_rg)
        if (
            not label
            or label in seen_shape_names
            or not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower <= 0.0
            or upper <= lower
        ):
            raise ValueError("targeted shape band is invalid")
        seen_shape_names.add(label)
        shell_index = _find_shape_shell(
            geometry,
            lower,
            upper,
            absolute_tolerance=edge_tolerance_rg,
            nominal_relative_tolerance=nominal_tolerance,
        )
        mask = geometry.cell_masks[shell_index]
        shell_weights = measures[mask] / float(np.sum(measures[mask]))
        centered = log_radius[mask] - float(
            shell_weights @ log_radius[mask]
        )
        radial_rms = float(
            np.sqrt(shell_weights @ np.square(centered))
        )
        if not np.isfinite(radial_rms) or radial_rms <= 0.0:
            raise ValueError(
                "targeted shape shell needs multiple distinct cell radii"
            )
        signed_weights = shell_weights * centered / radial_rms
        coefficients = np.zeros(n_cells, dtype=float)
        coefficients[mask] = signed_weights
        for component, field_name in (
            (3, "log_temperature"),
            (0, "log_surface_density"),
        ):
            names.append(f"shape_{label}_{field_name}_first")
            families.append("targeted_first_shape_moment")
            values.append(
                float(signed_weights @ state.primitives[mask, component])
            )
            scales.append(1.0)
            rows.append(
                _direct_primitive_row(
                    primitive_scales,
                    n_cells,
                    component,
                    coefficients,
                )
            )
    levels.append(
        _conditioned_level(
            name="plus_targeted_shape_moments",
            coordinate_names=names,
            coordinate_families=families,
            coordinate_values=values,
            coordinate_scales=scales,
            raw_rows=rows,
        )
    )

    interface_names = []
    interface_values = []
    interface_rows = []
    interface_scales = []
    physical_faces = C * state.weighted_face_fluxes_over_c
    physical_face_response = C * face_response
    component_flux_scales = {
        component: max(
            float(np.max(np.abs(physical_faces[:, component]))),
            np.finfo(float).tiny,
        )
        for component, _name in _INTERFACE_FLUX_COMPONENTS
    }
    for boundary_index, face in enumerate(
        geometry.edge_indices[1:-1],
        start=1,
    ):
        for component, component_name in _INTERFACE_FLUX_COMPONENTS:
            interface_names.append(
                f"interface_{boundary_index}_{component_name}"
            )
            interface_values.append(float(physical_faces[face, component]))
            interface_scales.append(component_flux_scales[component])
            interface_rows.append(physical_face_response[face, component])
    raw_interface = np.asarray(interface_rows, dtype=float)
    flux_scales = np.asarray(interface_scales, dtype=float)
    scaled_interface = raw_interface / flux_scales[:, None]
    return CausalFiveFieldMomentCoordinateLadder(
        geometry=geometry,
        levels=tuple(levels),
        storage_semantics=CAUSAL_WP10C8I_STORAGE_SEMANTICS,
        primitive_column_scales=primitive_scales,
        interface_flux_names=tuple(interface_names),
        interface_flux_values=np.asarray(interface_values, dtype=float),
        interface_flux_scales=flux_scales,
        raw_interface_flux_jacobian=raw_interface,
        interface_flux_jacobian=scaled_interface,
    )
