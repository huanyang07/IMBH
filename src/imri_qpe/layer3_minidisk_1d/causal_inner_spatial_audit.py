"""Nested-grid diagnostics for the causal five-field finite-volume system."""

from __future__ import annotations

import numpy as np
from scipy.sparse import diags, vstack
from scipy.sparse.linalg import splu

from imri_qpe.constants import C

from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    CausalFiveFieldDAEEvaluation,
    causal_five_field_colored_central_jacobian,
    causal_five_field_cell_states,
    causal_five_field_dae_jacobian_color_groups,
    causal_five_field_dae_jacobian_sparsity,
    causal_five_field_dae_scaling,
    causal_five_field_mapped_conserved_from_primitives,
    causal_five_field_path_temporal_storage_increment,
    causal_five_field_reduced_stationary_residual,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    pack_causal_five_field_state,
    unpack_causal_five_field_state,
)
from .causal_inner_evolution import causal_five_field_h_over_r_profile
from .causal_inner_geometry import KerrSchildColumnGrid


CAUSAL_FIVE_FIELD_NAMES = (
    "rest_mass",
    "radial_momentum",
    "angular_momentum",
    "killing_energy",
    "relaxing_stress",
)

CAUSAL_FIVE_FIELD_PRIMITIVE_NAMES = (
    "log_surface_density",
    "radial_velocity_over_c",
    "azimuthal_velocity_over_c",
    "log_temperature",
    "specific_stress",
)


def causal_nested_refinement_ratio(
    coarse_grid: KerrSchildColumnGrid,
    fine_grid: KerrSchildColumnGrid,
) -> int:
    """Return the exact integer refinement ratio for two nested grids."""

    coarse_edges = np.asarray(coarse_grid.edges, dtype=float)
    fine_edges = np.asarray(fine_grid.edges, dtype=float)
    n_coarse = coarse_edges.size - 1
    n_fine = fine_edges.size - 1
    if n_coarse < 1 or n_fine % n_coarse != 0:
        raise ValueError("causal grids do not have an integer refinement ratio")
    ratio = n_fine // n_coarse
    if ratio < 1 or not np.array_equal(coarse_edges, fine_edges[::ratio]):
        raise ValueError("causal grid faces are not exactly nested")
    fine_measures = np.asarray(fine_grid.cell_measures, dtype=float)
    coarse_measures = np.asarray(coarse_grid.cell_measures, dtype=float)
    restricted_measures = np.sum(
        fine_measures.reshape(n_coarse, ratio),
        axis=1,
    )
    if not np.allclose(
        restricted_measures,
        coarse_measures,
        rtol=2.0e-14,
        atol=0.0,
    ):
        raise ValueError("nested causal cell measures are inconsistent")
    return ratio


def causal_restrict_cell_integrals(
    coarse_grid: KerrSchildColumnGrid,
    fine_grid: KerrSchildColumnGrid,
    fine_integrals: np.ndarray,
) -> np.ndarray:
    """Sum fine cell integrals exactly onto the nested coarse cells."""

    ratio = causal_nested_refinement_ratio(coarse_grid, fine_grid)
    values = np.asarray(fine_integrals, dtype=float)
    n_fine = fine_grid.centers.size
    n_coarse = coarse_grid.centers.size
    if (
        values.ndim < 1
        or values.shape[0] != n_fine
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("fine cell integrals are invalid")
    reshaped = values.reshape((n_coarse, ratio) + values.shape[1:])
    return np.asarray(np.sum(reshaped, axis=1), dtype=float)


def causal_restrict_cell_averages(
    coarse_grid: KerrSchildColumnGrid,
    fine_grid: KerrSchildColumnGrid,
    fine_values: np.ndarray,
) -> np.ndarray:
    """Restrict fine cell densities with exact Kerr-Schild measures."""

    values = np.asarray(fine_values, dtype=float)
    fine_measures = np.asarray(fine_grid.cell_measures, dtype=float)
    weights = fine_measures.reshape(
        (fine_measures.size,) + (1,) * (values.ndim - 1)
    )
    restricted = causal_restrict_cell_integrals(
        coarse_grid,
        fine_grid,
        values * weights,
    )
    coarse_measures = np.asarray(coarse_grid.cell_measures, dtype=float)
    denominator = coarse_measures.reshape(
        (coarse_measures.size,) + (1,) * (values.ndim - 1)
    )
    return np.asarray(restricted / denominator, dtype=float)


def causal_coincident_fine_faces(
    coarse_grid: KerrSchildColumnGrid,
    fine_grid: KerrSchildColumnGrid,
    fine_face_values: np.ndarray,
) -> np.ndarray:
    """Return fine-grid values on faces shared exactly with the coarse grid."""

    ratio = causal_nested_refinement_ratio(coarse_grid, fine_grid)
    values = np.asarray(fine_face_values, dtype=float)
    if (
        values.ndim < 1
        or values.shape[0] != fine_grid.edges.size
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("fine face values are invalid")
    return np.asarray(values[::ratio], dtype=float)


def causal_spatial_difference_metrics(
    left: np.ndarray,
    right: np.ndarray,
    cell_measures: np.ndarray,
    radii: np.ndarray,
    *,
    exclude_boundary_cells: int = 0,
) -> dict:
    """Return measure-weighted norms and the peak location of one difference."""

    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    measures = np.asarray(cell_measures, dtype=float)
    radius = np.asarray(radii, dtype=float)
    if (
        first.shape != second.shape
        or first.ndim != 1
        or first.shape != measures.shape
        or first.shape != radius.shape
        or np.any(~np.isfinite(first))
        or np.any(~np.isfinite(second))
        or np.any(~np.isfinite(measures))
        or np.any(measures <= 0.0)
        or np.any(~np.isfinite(radius))
        or np.any(radius <= 0.0)
    ):
        raise ValueError("spatial comparison arrays are invalid")
    excluded = int(exclude_boundary_cells)
    if (
        excluded != exclude_boundary_cells
        or excluded < 0
        or 2 * excluded >= first.size
    ):
        raise ValueError("invalid number of excluded boundary cells")
    selected = slice(excluded, first.size - excluded or None)
    difference = first[selected] - second[selected]
    selected_measures = measures[selected]
    selected_radius = radius[selected]
    absolute = np.abs(difference)
    maximum_index = int(np.argmax(absolute))
    measure_sum = float(np.sum(selected_measures))
    amplitude = max(
        float(np.max(np.abs(first[selected]))),
        float(np.max(np.abs(second[selected]))),
        np.finfo(float).tiny,
    )
    return {
        "maximum_absolute_difference": float(absolute[maximum_index]),
        "measure_weighted_l1_difference": float(
            np.sum(selected_measures * absolute) / measure_sum
        ),
        "measure_weighted_l2_difference": float(
            np.sqrt(
                np.sum(selected_measures * difference**2) / measure_sum
            )
        ),
        "rms_difference": float(np.sqrt(np.mean(difference**2))),
        "maximum_difference_radius": float(
            selected_radius[maximum_index]
        ),
        "maximum_difference_relative_to_profile_amplitude": float(
            absolute[maximum_index] / amplitude
        ),
        "excluded_boundary_cells_per_side": excluded,
    }


def causal_spatial_contraction_order(
    coarse_pair_difference: float,
    fine_pair_difference: float,
) -> float:
    """Return the observed order from two successive mesh-pair differences."""

    coarse = float(coarse_pair_difference)
    fine = float(fine_pair_difference)
    if (
        not np.isfinite(coarse)
        or not np.isfinite(fine)
        or coarse <= 0.0
        or fine <= 0.0
    ):
        raise ValueError("spatial pair differences must be positive and finite")
    return float(np.log2(coarse / fine))


def causal_five_field_log_h_over_r_tangent(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    primitive_tangent: np.ndarray,
) -> np.ndarray:
    """Map one primitive tangent to the responsive-height logarithmic tangent."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(vector, n_cells)
    tangent = np.asarray(primitive_tangent, dtype=float)
    if tangent.shape != (n_cells, 5) or np.any(~np.isfinite(tangent)):
        raise ValueError("primitive tangent has the wrong shape or value")
    result = np.empty(n_cells, dtype=float)
    for index, radius in enumerate(context.grid.centers):
        sigma = float(np.exp(state.primitives[index, 0]))
        temperature = float(np.exp(state.primitives[index, 3]))
        derivatives = context.vertical_frequency.eos(
            float(radius)
        ).derivatives(sigma, temperature)
        result[index] = (
            derivatives.height_log_surface_density * tangent[index, 0]
            + derivatives.height_log_temperature * tangent[index, 3]
        )
    return result


def causal_five_field_profile_fields(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    evaluation: CausalFiveFieldDAEEvaluation | None = None,
) -> dict[str, np.ndarray]:
    """Return the primary and derived cell profiles used by spatial audits."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(vector, n_cells)
    cell_states = causal_five_field_cell_states(context, vector)
    if evaluation is None:
        evaluation = evaluate_causal_five_field_dae(vector, context)
    pressure = np.asarray(
        [cell.thermodynamics.integrated_pressure for cell in cell_states],
        dtype=float,
    )
    internal_energy = np.asarray(
        [
            cell.thermodynamics.specific_internal_energy
            for cell in cell_states
        ],
        dtype=float,
    )
    profiles = {
        name: np.asarray(state.primitives[:, index], dtype=float)
        for index, name in enumerate(CAUSAL_FIVE_FIELD_PRIMITIVE_NAMES)
    }
    profiles.update(
        {
            "log_h_over_r": np.log(
                causal_five_field_h_over_r_profile(context, vector)
            ),
            "log_integrated_pressure": np.log(pressure),
            "log_specific_internal_energy": np.log(internal_energy),
            "scattering_optical_depth": np.asarray(
                evaluation.scattering_optical_depths,
                dtype=float,
            ),
        }
    )
    return profiles


def causal_five_field_residual_terms(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    evaluation: CausalFiveFieldDAEEvaluation,
) -> dict[str, np.ndarray]:
    """Return signed blocks whose sum reconstructs the conservation rows."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(vector, n_cells)
    central = np.asarray(
        evaluation.central_weighted_face_fluxes_over_c,
        dtype=float,
    )
    dissipation = np.asarray(
        evaluation.rusanov_dissipation_weighted_face_fluxes_over_c,
        dtype=float,
    )
    numerical = np.asarray(
        evaluation.numerical_weighted_face_fluxes_over_c,
        dtype=float,
    )
    closure = (
        np.asarray(state.weighted_face_fluxes_over_c, dtype=float)
        - numerical
    )
    vertical = np.zeros((n_cells, 5), dtype=float)
    vertical[:, :4] = np.asarray(
        evaluation.temporal_vertical_storage,
        dtype=float,
    )
    terms = {
        "temporal_conserved_storage": np.asarray(
            evaluation.temporal_conserved_storage,
            dtype=float,
        ),
        "temporal_vertical_storage": vertical,
        "central_face_transport": (
            central[1:] - central[:-1]
        ),
        "rusanov_face_transport": (
            dissipation[1:] - dissipation[:-1]
        ),
        "flux_primary_closure": closure[1:] - closure[:-1],
    }
    terms.update(
        {
            name: -np.asarray(values, dtype=float)
            for name, values in (
                evaluation.integrated_source_components_per_ct.items()
            )
        }
    )
    return terms


def causal_five_field_term_reconstruction_defect(
    evaluation: CausalFiveFieldDAEEvaluation,
    terms: dict[str, np.ndarray],
) -> dict[str, float]:
    """Audit reconstruction of the five conservation rows from signed terms."""

    values = tuple(np.asarray(value, dtype=float) for value in terms.values())
    if not values:
        raise ValueError("at least one residual term is required")
    shape = np.asarray(evaluation.conservation_rows, dtype=float).shape
    if any(value.shape != shape for value in values):
        raise ValueError("residual term shape does not match conservation rows")
    reconstructed = np.sum(np.asarray(values, dtype=float), axis=0)
    target = np.asarray(evaluation.conservation_rows, dtype=float)
    absolute = np.abs(reconstructed - target)
    scale = np.maximum(
        np.abs(target),
        np.sum(np.abs(np.asarray(values, dtype=float)), axis=0),
    )
    scale = np.maximum(scale, 1.0)
    return {
        "maximum_absolute_defect": float(np.max(absolute)),
        "maximum_relative_defect": float(np.max(absolute / scale)),
    }


def causal_five_field_constraint_manifold_jvp(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    primitive_direction: np.ndarray,
    *,
    finite_difference_step: float = 2.0e-6,
) -> dict:
    """Differentiate spatial residual terms along one primitive direction."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(vector, n_cells)
    direction = np.asarray(primitive_direction, dtype=float)
    step = float(finite_difference_step)
    if (
        direction.shape != (n_cells, 5)
        or np.any(~np.isfinite(direction))
        or not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("constraint-manifold JVP inputs are invalid")

    evaluations = []
    terms = []
    for sign in (1.0, -1.0):
        trial_state = causal_five_field_state_from_primitives(
            context,
            state.primitives + sign * step * direction,
        )
        trial_vector = pack_causal_five_field_state(trial_state)
        trial_evaluation = evaluate_causal_five_field_dae(
            trial_vector,
            context,
        )
        evaluations.append(trial_evaluation)
        terms.append(
            causal_five_field_residual_terms(
                context,
                trial_vector,
                trial_evaluation,
            )
        )
    denominator = 2.0 * step
    term_names = tuple(terms[0])
    if tuple(terms[1]) != term_names:
        raise RuntimeError("constraint-manifold JVP term schemas differ")
    term_jvps = {
        name: (
            np.asarray(terms[0][name], dtype=float)
            - np.asarray(terms[1][name], dtype=float)
        )
        / denominator
        for name in term_names
    }
    conservation_jvp = (
        np.asarray(evaluations[0].conservation_rows, dtype=float)
        - np.asarray(evaluations[1].conservation_rows, dtype=float)
    ) / denominator
    reconstructed = np.sum(
        np.asarray(list(term_jvps.values()), dtype=float),
        axis=0,
    )
    scale = np.maximum(
        np.abs(conservation_jvp),
        np.sum(np.abs(np.asarray(list(term_jvps.values()))), axis=0),
    )
    scale = np.maximum(scale, 1.0)
    absolute_defect = np.abs(reconstructed - conservation_jvp)
    global_scale = max(float(np.max(scale)), 1.0)
    return {
        "conservation_jvp": conservation_jvp,
        "term_jvps": term_jvps,
        "maximum_reconstruction_relative_defect": float(
            np.max(absolute_defect) / global_scale
        ),
        "maximum_entrywise_reconstruction_relative_defect": float(
            np.max(absolute_defect / scale)
        ),
        "maximum_reconstruction_absolute_defect": float(
            np.max(absolute_defect)
        ),
        "finite_difference_step": step,
    }


def _causal_five_field_scaled_stationary_and_be_jacobians(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    *,
    finite_difference_step: float,
    descriptor_timestep_seconds: float,
):
    """Build consistently scaled stationary and backward-Euler Jacobians."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(vector, n_cells)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    timestep = float(descriptor_timestep_seconds)
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("descriptor timestep must be positive and finite")
    zero = np.zeros_like(np.asarray(vector, dtype=float))
    pattern = causal_five_field_dae_jacobian_sparsity(
        n_cells,
        spatial_reconstruction=context.spatial_reconstruction,
        boundary_trace_reconstruction=(
            context.boundary_trace_reconstruction
        ),
        cell_rate_scheme=context.cell_rate_scheme,
        cell_source_quadrature=context.cell_source_quadrature,
        cell_storage_quadrature=context.cell_storage_quadrature,
    )

    def scaled_residual(
        scaled_increment: np.ndarray,
        *,
        backward_euler: bool,
    ) -> np.ndarray:
        trial = (
            np.asarray(vector, dtype=float)
            + np.asarray(scaling.column_scales, dtype=float)
            * np.asarray(scaled_increment, dtype=float)
        )
        if backward_euler:
            trial_evaluation = evaluate_causal_five_field_dae(
                trial,
                context,
                old_vector=np.asarray(vector, dtype=float),
                timestep_seconds=timestep,
            )
        else:
            trial_evaluation = evaluate_causal_five_field_dae(
                trial,
                context,
            )
        return (
            np.asarray(trial_evaluation.residual, dtype=float)
            / np.asarray(scaling.row_scales, dtype=float)
        )

    stationary = causal_five_field_colored_central_jacobian(
        lambda increment: scaled_residual(
            increment,
            backward_euler=False,
        ),
        zero,
        pattern,
        finite_difference_step=finite_difference_step,
    )
    backward_euler = causal_five_field_colored_central_jacobian(
        lambda increment: scaled_residual(
            increment,
            backward_euler=True,
        ),
        zero,
        pattern,
        finite_difference_step=finite_difference_step,
    )
    return state, evaluation, scaling, stationary, backward_euler


def _equilibrated_sparse_multiple_solve(
    matrix,
    right_hand_sides: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Solve a sparse square system for one or more dense right-hand sides."""

    sparse = matrix.tocsr().astype(float)
    right = np.asarray(right_hand_sides, dtype=float)
    if right.ndim == 1:
        right = right[:, None]
    if (
        sparse.shape[0] != sparse.shape[1]
        or right.ndim != 2
        or right.shape[0] != sparse.shape[0]
        or np.any(~np.isfinite(sparse.data))
        or np.any(~np.isfinite(right))
    ):
        raise ValueError("sparse multiple-right-hand-side system is invalid")
    tiny = np.finfo(float).tiny
    row_maximum = np.asarray(
        np.abs(sparse).max(axis=1).toarray(),
        dtype=float,
    ).ravel()
    if np.any(row_maximum <= tiny):
        raise np.linalg.LinAlgError("sparse matrix has a zero row")
    row_scale = 1.0 / row_maximum
    row_scaled = diags(row_scale) @ sparse
    column_maximum = np.asarray(
        np.abs(row_scaled).max(axis=0).toarray(),
        dtype=float,
    ).ravel()
    if np.any(column_maximum <= tiny):
        raise np.linalg.LinAlgError("sparse matrix has a zero column")
    column_scale = 1.0 / column_maximum
    balanced = (
        diags(row_scale)
        @ sparse
        @ diags(column_scale)
    ).tocsc()
    factorization = splu(balanced)
    balanced_solution = factorization.solve(
        row_scale[:, None] * right
    )
    solution = column_scale[:, None] * balanced_solution
    residual = sparse @ solution - right
    relative = float(
        np.max(np.abs(residual))
        / max(float(np.max(np.abs(right))), tiny)
    )
    return np.asarray(solution, dtype=float), relative


def _causal_five_field_reduce_stationary_sparse_jacobian(
    stationary,
    scaling,
    n_cells: int,
) -> dict:
    """Eliminate conserved and face variables from one stationary Jacobian."""

    n_reduced = 5 * int(n_cells)
    total = int(stationary.shape[0])
    if stationary.shape[1] != total or total <= 2 * n_reduced:
        raise ValueError("stationary Jacobian dimensions are invalid")
    conservation_rows = np.arange(n_reduced)
    algebraic_rows = np.arange(n_reduced, total)
    conserved_columns = np.arange(n_reduced)
    primitive_columns = np.arange(n_reduced, 2 * n_reduced)
    face_columns = np.arange(2 * n_reduced, total)
    algebraic_columns = np.concatenate(
        (conserved_columns, face_columns)
    )

    algebraic_block = stationary[algebraic_rows][:, algebraic_columns]
    algebraic_from_primitive = stationary[algebraic_rows][
        :,
        primitive_columns,
    ]
    solved, solve_defect = _equilibrated_sparse_multiple_solve(
        algebraic_block,
        algebraic_from_primitive.toarray(),
    )
    algebraic_response = -solved
    reduced = (
        stationary[conservation_rows][:, primitive_columns].toarray()
        + stationary[conservation_rows][:, algebraic_columns]
        @ algebraic_response
    )
    algebraic_reconstruction = (
        algebraic_block @ algebraic_response
        + algebraic_from_primitive
    )
    algebraic_scale = max(
        float(np.max(np.abs(algebraic_from_primitive.data)))
        if algebraic_from_primitive.nnz
        else 0.0,
        1.0,
    )
    return {
        "stationary_reduced_scaled_jacobian": np.asarray(
            reduced,
            dtype=float,
        ),
        "algebraic_response_scaled": np.asarray(
            algebraic_response,
            dtype=float,
        ),
        "primitive_column_scales": np.asarray(
            scaling.column_scales[primitive_columns],
            dtype=float,
        ),
        "algebraic_column_scales": np.asarray(
            scaling.column_scales[algebraic_columns],
            dtype=float,
        ),
        "conservation_row_scales": np.asarray(
            scaling.row_scales[conservation_rows],
            dtype=float,
        ),
        "algebraic_columns": algebraic_columns,
        "conserved_algebraic_rows": np.arange(n_reduced),
        "face_algebraic_rows": np.arange(
            n_reduced,
            algebraic_columns.size,
        ),
        "dimensions": (n_reduced, n_reduced),
        "full_dimensions": stationary.shape,
        "stationary_nonzeros": int(stationary.nnz),
        "algebraic_solve_relative_defect": solve_defect,
        "maximum_scaled_algebraic_reconstruction_defect": float(
            np.max(np.abs(algebraic_reconstruction)) / algebraic_scale
        ),
    }


def causal_five_field_reduced_stationary_jacobian(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    *,
    finite_difference_step: float = 2.0e-6,
) -> dict:
    """Build only the Schur-reduced stationary primitive Jacobian.

    Unlike :func:`causal_five_field_reduced_descriptor_matrices`, this
    diagnostic does not construct a backward-Euler Jacobian or a storage
    matrix.  It therefore permits the stationary finite-difference step to be
    scanned while independently computed storage blocks remain fixed.
    """

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    values = np.asarray(vector, dtype=float)
    step = float(finite_difference_step)
    if (
        values.shape != (15 * n_cells + 5,)
        or np.any(~np.isfinite(values))
        or not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("stationary-Jacobian inputs are invalid")
    state = unpack_causal_five_field_state(values, n_cells)
    evaluation = evaluate_causal_five_field_dae(values, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    zero = np.zeros_like(values)
    pattern = causal_five_field_dae_jacobian_sparsity(
        n_cells,
        spatial_reconstruction=context.spatial_reconstruction,
        boundary_trace_reconstruction=(
            context.boundary_trace_reconstruction
        ),
        cell_rate_scheme=context.cell_rate_scheme,
        cell_source_quadrature=context.cell_source_quadrature,
        cell_storage_quadrature=context.cell_storage_quadrature,
    )

    def scaled_stationary_residual(
        scaled_increment: np.ndarray,
    ) -> np.ndarray:
        trial = (
            values
            + np.asarray(scaling.column_scales, dtype=float)
            * np.asarray(scaled_increment, dtype=float)
        )
        trial_evaluation = evaluate_causal_five_field_dae(
            trial,
            context,
        )
        return (
            np.asarray(trial_evaluation.residual, dtype=float)
            / np.asarray(scaling.row_scales, dtype=float)
        )

    stationary = causal_five_field_colored_central_jacobian(
        scaled_stationary_residual,
        zero,
        pattern,
        finite_difference_step=step,
    )
    result = _causal_five_field_reduce_stationary_sparse_jacobian(
        stationary,
        scaling,
        n_cells,
    )
    result.update(
        {
            "finite_difference_step": step,
            "stationary_jacobian_source": (
                "independent_full_dae_colored_schur"
            ),
        }
    )
    return result


def causal_five_field_reduced_descriptor_matrices(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    *,
    finite_difference_step: float = 2.0e-6,
    descriptor_timestep_seconds: float = 1.0,
) -> dict:
    """Eliminate algebraic variables from the finite causal descriptor.

    The returned dense matrices act on scaled primitive perturbations.  They
    satisfy ``M dp/dt + K dp = 0`` for the frozen-coefficient linearized DAE;
    infinite algebraic modes have already been removed by the exact Schur
    response.
    """

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    timestep = float(descriptor_timestep_seconds)
    (
        _state,
        _evaluation,
        scaling,
        stationary,
        backward_euler,
    ) = _causal_five_field_scaled_stationary_and_be_jacobians(
        context,
        vector,
        finite_difference_step=finite_difference_step,
        descriptor_timestep_seconds=timestep,
    )
    n_reduced = 5 * n_cells
    stationary_reduction = (
        _causal_five_field_reduce_stationary_sparse_jacobian(
            stationary,
            scaling,
            n_cells,
        )
    )
    algebraic_response = stationary_reduction[
        "algebraic_response_scaled"
    ]
    algebraic_columns = stationary_reduction["algebraic_columns"]
    conservation_rows = np.arange(n_reduced)
    algebraic_rows = np.arange(n_reduced, stationary.shape[0])
    primitive_columns = np.arange(n_reduced, 2 * n_reduced)
    descriptor = timestep * (backward_euler - stationary)
    descriptor_reduced = (
        descriptor[conservation_rows][:, primitive_columns].toarray()
        + descriptor[conservation_rows][:, algebraic_columns]
        @ algebraic_response
    )
    descriptor_algebraic = descriptor[algebraic_rows]
    descriptor_scale = max(
        float(np.max(np.abs(descriptor.data)))
        if descriptor.nnz
        else 0.0,
        1.0,
    )
    return {
        **stationary_reduction,
        "descriptor_reduced_scaled_matrix": np.asarray(
            descriptor_reduced,
            dtype=float,
        ),
        "descriptor_nonzeros": int(descriptor.nnz),
        "maximum_scaled_descriptor_algebraic_row": float(
            (
                np.max(np.abs(descriptor_algebraic.data))
                if descriptor_algebraic.nnz
                else 0.0
            )
            / descriptor_scale
        ),
        "finite_difference_step": float(finite_difference_step),
        "descriptor_timestep_seconds": timestep,
    }


def _causal_five_field_reduced_storage_action_with_scale(
    context: CausalFiveFieldDAEContext,
    primitive_vector: np.ndarray,
    primitive_rate_per_s: np.ndarray,
    primitive_scales: np.ndarray,
    *,
    storage_difference_step: float,
    storage_quadrature_order: int,
    storage_directional_step: float,
    conserved_difference_order: int = 2,
    include_conserved: bool = True,
) -> dict:
    """Apply the nonlinear primitive storage one-form to one rate.

    The centered path construction differentiates the complete temporal
    storage, including all four implemented responsive-height components.
    It therefore avoids treating height work as an energy-only state
    function.
    """

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    primitives = np.asarray(primitive_vector, dtype=float)
    rate = np.asarray(primitive_rate_per_s, dtype=float)
    scales = np.asarray(primitive_scales, dtype=float)
    expected = (5 * n_cells,)
    if (
        primitives.shape != expected
        or rate.shape != expected
        or scales.shape != expected
        or np.any(~np.isfinite(primitives))
        or np.any(~np.isfinite(rate))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError("reduced storage-action inputs are invalid")
    difference_step = float(storage_difference_step)
    if not np.isfinite(difference_step) or difference_step <= 0.0:
        raise ValueError(
            "storage_difference_step must be positive and finite"
        )
    difference_order = int(conserved_difference_order)
    if difference_order not in (2, 4, 6):
        raise ValueError(
            "conserved_difference_order must be two, four, or six"
        )
    scaled_rate = rate / scales
    maximum_scaled_rate = float(np.max(np.abs(scaled_rate)))
    zeros = np.zeros((n_cells, 5), dtype=float)
    if maximum_scaled_rate == 0.0:
        return {
            "total_conservation_storage_per_ct": zeros,
            "conserved_storage_per_ct": np.array(zeros, copy=True),
            "vertical_storage_per_ct": np.array(zeros, copy=True),
            "storage_timestep_seconds": 0.0,
            "maximum_scaled_primitive_increment": 0.0,
            "conserved_difference_order": difference_order,
        }

    storage_timestep = difference_step / maximum_scaled_rate
    old = primitives.reshape(n_cells, 5)
    primitive_increment = (
        storage_timestep * rate.reshape(n_cells, 5)
    )
    if include_conserved:
        plus_conserved = (
            causal_five_field_mapped_conserved_from_primitives(
                context,
                old + primitive_increment,
            )
        )
        minus_conserved = (
            causal_five_field_mapped_conserved_from_primitives(
                context,
                old - primitive_increment,
            )
        )
        if difference_order == 4:
            plus_two_conserved = (
                causal_five_field_mapped_conserved_from_primitives(
                    context,
                    old + 2.0 * primitive_increment,
                )
            )
            minus_two_conserved = (
                causal_five_field_mapped_conserved_from_primitives(
                    context,
                    old - 2.0 * primitive_increment,
                )
            )
        elif difference_order == 6:
            plus_two_conserved = (
                causal_five_field_mapped_conserved_from_primitives(
                    context,
                    old + 2.0 * primitive_increment,
                )
            )
            minus_two_conserved = (
                causal_five_field_mapped_conserved_from_primitives(
                    context,
                    old - 2.0 * primitive_increment,
                )
            )
            plus_three_conserved = (
                causal_five_field_mapped_conserved_from_primitives(
                    context,
                    old + 3.0 * primitive_increment,
                )
            )
            minus_three_conserved = (
                causal_five_field_mapped_conserved_from_primitives(
                    context,
                    old - 3.0 * primitive_increment,
                )
            )
    plus = causal_five_field_path_temporal_storage_increment(
        context,
        old,
        old + primitive_increment,
        quadrature_order=storage_quadrature_order,
        directional_step=storage_directional_step,
    )
    minus = causal_five_field_path_temporal_storage_increment(
        context,
        old,
        old - primitive_increment,
        quadrature_order=storage_quadrature_order,
        directional_step=storage_directional_step,
    )
    denominator = 2.0 * C * storage_timestep
    measures = np.asarray(context.grid.cell_measures, dtype=float)[:, None]
    if include_conserved:
        if difference_order == 2:
            conserved = measures * (
                np.asarray(plus_conserved, dtype=float)
                - np.asarray(minus_conserved, dtype=float)
            ) / denominator
        elif difference_order == 4:
            conserved = measures * (
                -np.asarray(plus_two_conserved, dtype=float)
                + 8.0 * np.asarray(plus_conserved, dtype=float)
                - 8.0 * np.asarray(minus_conserved, dtype=float)
                + np.asarray(minus_two_conserved, dtype=float)
            ) / (12.0 * C * storage_timestep)
        else:
            conserved = measures * (
                -np.asarray(minus_three_conserved, dtype=float)
                + 9.0 * np.asarray(minus_two_conserved, dtype=float)
                - 45.0 * np.asarray(minus_conserved, dtype=float)
                + 45.0 * np.asarray(plus_conserved, dtype=float)
                - 9.0 * np.asarray(plus_two_conserved, dtype=float)
                + np.asarray(plus_three_conserved, dtype=float)
            ) / (60.0 * C * storage_timestep)
    else:
        conserved = np.zeros((n_cells, 5), dtype=float)
    vertical = np.zeros((n_cells, 5), dtype=float)
    vertical[:, :4] = measures * (
        np.asarray(plus.vertical_killing_increment, dtype=float)
        - np.asarray(minus.vertical_killing_increment, dtype=float)
    ) / denominator
    return {
        "total_conservation_storage_per_ct": conserved + vertical,
        "conserved_storage_per_ct": conserved,
        "vertical_storage_per_ct": vertical,
        "storage_timestep_seconds": float(storage_timestep),
        "maximum_scaled_primitive_increment": difference_step,
        "conserved_difference_order": difference_order,
    }


def causal_five_field_reduced_storage_action(
    context: CausalFiveFieldDAEContext,
    primitive_vector: np.ndarray,
    primitive_rate_per_s: np.ndarray,
    *,
    storage_difference_step: float = 1.0e-4,
    storage_quadrature_order: int = 4,
    storage_directional_step: float = 1.0e-3,
    conserved_difference_order: int = 2,
) -> dict:
    """Apply the complete reduced temporal storage to a primitive rate.

    Inputs and the returned rate use physical primitive coordinates and
    coordinate seconds.  Responsive-height storage is returned separately
    as the vector ``(0, P_R, J, E_K, 0)`` contribution as well as in the
    total action.
    """

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    primitives = np.asarray(primitive_vector, dtype=float)
    if primitives.shape != (5 * n_cells,):
        raise ValueError("reduced primitive vector has the wrong shape")
    charts = primitives.reshape(n_cells, 5)
    primitive_scales = np.ones_like(charts)
    stress_magnitude = np.abs(charts[:, 4])
    primitive_scales[:, 4] = np.maximum(
        stress_magnitude,
        max(float(np.median(stress_magnitude)), 1.0e-14),
    )
    return _causal_five_field_reduced_storage_action_with_scale(
        context,
        primitives,
        primitive_rate_per_s,
        primitive_scales.ravel(),
        storage_difference_step=storage_difference_step,
        storage_quadrature_order=storage_quadrature_order,
        storage_directional_step=storage_directional_step,
        conserved_difference_order=conserved_difference_order,
    )


def _causal_five_field_reduced_storage_pattern(
    context: CausalFiveFieldDAEContext,
):
    """Return primitive storage dependencies after exact map elimination."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    n_reduced = 5 * n_cells
    full_pattern = causal_five_field_dae_jacobian_sparsity(
        n_cells,
        spatial_reconstruction=context.spatial_reconstruction,
        boundary_trace_reconstruction=(
            context.boundary_trace_reconstruction
        ),
        cell_rate_scheme=context.cell_rate_scheme,
        cell_source_quadrature=context.cell_source_quadrature,
        cell_storage_quadrature=context.cell_storage_quadrature,
    )
    return full_pattern[
        n_reduced : 2 * n_reduced,
        n_reduced : 2 * n_reduced,
    ].tocsr()


def _causal_five_field_colored_component_jacobians(
    residual_components,
    values: np.ndarray,
    pattern,
    *,
    finite_difference_step: float | np.ndarray,
) -> tuple[dict[str, np.ndarray], int]:
    """Differentiate several residual blocks with shared colored evaluations."""

    base = np.asarray(values, dtype=float)
    declared = pattern.tocsc()
    if (
        declared.shape != (base.size, base.size)
        or np.any(~np.isfinite(base))
    ):
        raise ValueError("colored component values or pattern are invalid")
    raw_step = np.asarray(finite_difference_step, dtype=float)
    if raw_step.ndim == 0:
        steps = np.full(base.size, float(raw_step), dtype=float)
    elif raw_step.shape == (base.size,):
        steps = np.array(raw_step, copy=True)
    else:
        raise ValueError("component finite-difference steps have wrong shape")
    if np.any(~np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("component finite-difference step is invalid")
    groups = causal_five_field_dae_jacobian_color_groups(declared)
    jacobians: dict[str, np.ndarray] | None = None
    component_names: tuple[str, ...] | None = None
    for group in groups:
        plus = np.array(base, copy=True)
        minus = np.array(base, copy=True)
        plus[group] += steps[group]
        minus[group] -= steps[group]
        plus_components = {
            name: np.asarray(component, dtype=float)
            for name, component in residual_components(plus).items()
        }
        minus_components = {
            name: np.asarray(component, dtype=float)
            for name, component in residual_components(minus).items()
        }
        if component_names is None:
            component_names = tuple(plus_components)
            if tuple(minus_components) != component_names:
                raise ValueError("colored component schemas differ")
            if any(
                plus_components[name].shape != (base.size,)
                or minus_components[name].shape != (base.size,)
                for name in component_names
            ):
                raise ValueError("colored component residual shape is invalid")
            jacobians = {
                name: np.zeros(declared.shape, dtype=float)
                for name in component_names
            }
        elif (
            tuple(plus_components) != component_names
            or tuple(minus_components) != component_names
        ):
            raise ValueError("colored component schemas changed")
        assert jacobians is not None
        assert component_names is not None
        for column in group:
            start = declared.indptr[column]
            stop = declared.indptr[column + 1]
            rows = declared.indices[start:stop]
            for name in component_names:
                jacobians[name][rows, column] = (
                    plus_components[name][rows]
                    - minus_components[name][rows]
                ) / (2.0 * steps[column])
    if jacobians is None:
        raise ValueError("storage pattern has no color groups")
    return jacobians, len(groups)


def _causal_five_field_reduced_storage_matrix_with_scale(
    context: CausalFiveFieldDAEContext,
    primitive_vector: np.ndarray,
    primitive_scales: np.ndarray,
    conservation_scales: np.ndarray,
    *,
    finite_difference_step: float,
    storage_quadrature_order: int,
    storage_directional_step: float,
    include_vertical: bool = True,
) -> tuple[dict[str, np.ndarray], int]:
    """Build the reduced storage matrix on its declared mapped stencil."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    n_reduced = 5 * n_cells
    primitives = np.asarray(primitive_vector, dtype=float)
    primitive_scale = np.asarray(primitive_scales, dtype=float)
    row_scale = np.asarray(conservation_scales, dtype=float)
    if (
        primitives.shape != (n_reduced,)
        or primitive_scale.shape != (n_reduced,)
        or row_scale.shape != (n_reduced,)
    ):
        raise ValueError("reduced storage-matrix inputs have wrong shapes")
    storage_pattern = _causal_five_field_reduced_storage_pattern(context)
    base = primitives.reshape(n_cells, 5)
    base_mapped = causal_five_field_mapped_conserved_from_primitives(
        context,
        base,
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)[:, None]

    def scaled_storage_increment_components(
        scaled_increment: np.ndarray,
    ) -> dict[str, np.ndarray]:
        new = (
            primitives
            + primitive_scale * np.asarray(scaled_increment, dtype=float)
        ).reshape(n_cells, 5)
        new_mapped = causal_five_field_mapped_conserved_from_primitives(
            context,
            new,
        )
        conserved = (
            measures
            * np.asarray(new_mapped - base_mapped, dtype=float)
            / C
            / row_scale.reshape(n_cells, 5)
        )
        vertical = np.zeros((n_cells, 5), dtype=float)
        if include_vertical:
            path = causal_five_field_path_temporal_storage_increment(
                context,
                base,
                new,
                quadrature_order=storage_quadrature_order,
                directional_step=storage_directional_step,
            )
            vertical[:, :4] = (
                measures
                * np.asarray(
                    path.vertical_killing_increment,
                    dtype=float,
                )
                / C
                / row_scale.reshape(n_cells, 5)[:, :4]
            )
        return {
            "conserved": conserved.ravel(),
            "vertical": vertical.ravel(),
            "total": (conserved + vertical).ravel(),
        }

    matrices, color_count = (
        _causal_five_field_colored_component_jacobians(
            scaled_storage_increment_components,
            np.zeros(n_reduced, dtype=float),
            storage_pattern,
            finite_difference_step=finite_difference_step,
        )
    )
    matrices["total"] = matrices["conserved"] + matrices["vertical"]
    return matrices, color_count


def causal_five_field_reduced_storage_matrices(
    context: CausalFiveFieldDAEContext,
    primitive_vector: np.ndarray,
    *,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    finite_difference_step: float = 2.0e-6,
    storage_quadrature_order: int = 4,
    storage_directional_step: float = 1.0e-3,
) -> dict:
    """Build independent mapped and responsive-height storage matrices.

    The matrices act from fixed scaled primitive coordinates to fixed scaled
    conservation rows.  The mapped-conserved and responsive-height pieces are
    returned separately so their inner finite-difference convergence can be
    audited without rebuilding a stationary Jacobian.
    """

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    n_reduced = 5 * n_cells
    primitives = np.asarray(primitive_vector, dtype=float)
    primitive_scales = np.asarray(
        primitive_column_scales,
        dtype=float,
    )
    conservation_scales = np.asarray(
        conservation_row_scales,
        dtype=float,
    )
    step = float(finite_difference_step)
    if (
        primitives.shape != (n_reduced,)
        or primitive_scales.shape != (n_reduced,)
        or conservation_scales.shape != (n_reduced,)
        or np.any(~np.isfinite(primitives))
        or np.any(~np.isfinite(primitive_scales))
        or np.any(~np.isfinite(conservation_scales))
        or np.any(primitive_scales <= 0.0)
        or np.any(conservation_scales <= 0.0)
        or not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("reduced storage-matrix inputs are invalid")
    matrices, color_count = (
        _causal_five_field_reduced_storage_matrix_with_scale(
            context,
            primitives,
            primitive_scales,
            conservation_scales,
            finite_difference_step=step,
            storage_quadrature_order=storage_quadrature_order,
            storage_directional_step=storage_directional_step,
        )
    )
    component_defect = matrices["total"] - (
        matrices["conserved"] + matrices["vertical"]
    )
    return {
        "descriptor_reduced_scaled_matrix": matrices["total"],
        "conserved_descriptor_reduced_scaled_matrix": (
            matrices["conserved"]
        ),
        "vertical_descriptor_reduced_scaled_matrix": (
            matrices["vertical"]
        ),
        "primitive_column_scales": primitive_scales,
        "conservation_row_scales": conservation_scales,
        "finite_difference_step": step,
        "storage_quadrature_order": int(storage_quadrature_order),
        "storage_directional_step": float(storage_directional_step),
        "storage_component_colors": color_count,
        "storage_component_paired_evaluations": 2 * color_count,
        "maximum_scaled_component_reconstruction_defect": float(
            np.max(np.abs(component_defect))
        ),
        "mass_matrix_source": (
            "independent_gauss_mapped_vector_storage_one_form"
        ),
    }


def causal_five_field_reduced_storage_rate_derivatives(
    context: CausalFiveFieldDAEContext,
    primitive_vector: np.ndarray,
    primitive_rate_per_s: np.ndarray,
    *,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    storage_matrix_difference_step: float = 2.0e-6,
    storage_rate_derivative_step: float,
    storage_difference_step: float = 1.0e-4,
    storage_quadrature_order: int = 4,
    storage_directional_step: float = 1.0e-3,
    conserved_difference_order: int = 2,
    backend: str = "nested_matrix",
) -> dict:
    """Build only ``DM[., p_dot]`` in fixed scaled coordinates.

    ``backend="nested_matrix"`` preserves the production construction: it
    builds the mapped-conserved storage matrix at every outer state and then
    contracts that matrix with ``p_dot``.  Its inner mapped-storage step and
    outer state-derivative step are independent inputs.

    ``backend="direct_action"`` is an independent diagnostic construction.
    It applies the complete nonlinear conserved-plus-responsive-height
    storage one-form directly to ``p_dot`` and differentiates that action only
    once in the outer state coordinate.  It therefore avoids differentiating
    a numerically differentiated mass matrix.  The production default and
    its numerical path remain unchanged.
    """

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    n_reduced = 5 * n_cells
    primitives = np.asarray(primitive_vector, dtype=float)
    physical_rate = np.asarray(primitive_rate_per_s, dtype=float)
    if physical_rate.shape == (n_cells, 5):
        physical_rate = physical_rate.ravel()
    primitive_scales = np.asarray(
        primitive_column_scales,
        dtype=float,
    )
    conservation_scales = np.asarray(
        conservation_row_scales,
        dtype=float,
    )
    inner_step = float(storage_matrix_difference_step)
    outer_step = float(storage_rate_derivative_step)
    derivative_backend = str(backend)
    if (
        primitives.shape != (n_reduced,)
        or physical_rate.shape != (n_reduced,)
        or primitive_scales.shape != (n_reduced,)
        or conservation_scales.shape != (n_reduced,)
        or np.any(~np.isfinite(primitives))
        or np.any(~np.isfinite(physical_rate))
        or np.any(~np.isfinite(primitive_scales))
        or np.any(~np.isfinite(conservation_scales))
        or np.any(primitive_scales <= 0.0)
        or np.any(conservation_scales <= 0.0)
        or not np.isfinite(inner_step)
        or inner_step <= 0.0
        or not np.isfinite(outer_step)
        or outer_step <= 0.0
    ):
        raise ValueError("storage-rate derivative inputs are invalid")
    if derivative_backend not in ("nested_matrix", "direct_action"):
        raise ValueError("storage-rate derivative backend is unsupported")
    scaled_rate = physical_rate / primitive_scales
    storage_pattern = _causal_five_field_reduced_storage_pattern(context)
    storage_color_count = len(
        causal_five_field_dae_jacobian_color_groups(storage_pattern)
    )
    nested_storage_color_counts: list[int] = []

    def nested_storage_rate_components_at_increment(
        scaled_increment: np.ndarray,
    ) -> dict[str, np.ndarray]:
        perturbed_primitives = (
            primitives
            + primitive_scales
            * np.asarray(scaled_increment, dtype=float)
        )
        component_matrices, nested_color_count = (
            _causal_five_field_reduced_storage_matrix_with_scale(
                context,
                perturbed_primitives,
                primitive_scales,
                conservation_scales,
                finite_difference_step=inner_step,
                storage_quadrature_order=storage_quadrature_order,
                storage_directional_step=storage_directional_step,
                include_vertical=False,
            )
        )
        nested_storage_color_counts.append(nested_color_count)
        conserved = component_matrices["conserved"] @ scaled_rate
        vertical_action = (
            _causal_five_field_reduced_storage_action_with_scale(
                context,
                perturbed_primitives,
                physical_rate,
                primitive_scales,
                storage_difference_step=storage_difference_step,
                storage_quadrature_order=storage_quadrature_order,
                storage_directional_step=storage_directional_step,
                include_conserved=False,
            )
        )
        vertical = (
            np.asarray(
                vertical_action["vertical_storage_per_ct"],
                dtype=float,
            ).ravel()
            / conservation_scales
        )
        return {
            "conserved": conserved,
            "vertical": vertical,
            "total": conserved + vertical,
        }

    def direct_storage_rate_components_at_increment(
        scaled_increment: np.ndarray,
    ) -> dict[str, np.ndarray]:
        perturbed_primitives = (
            primitives
            + primitive_scales
            * np.asarray(scaled_increment, dtype=float)
        )
        action = _causal_five_field_reduced_storage_action_with_scale(
            context,
            perturbed_primitives,
            physical_rate,
            primitive_scales,
            storage_difference_step=storage_difference_step,
            storage_quadrature_order=storage_quadrature_order,
            storage_directional_step=storage_directional_step,
            conserved_difference_order=conserved_difference_order,
            include_conserved=True,
        )
        conserved = (
            np.asarray(
                action["conserved_storage_per_ct"],
                dtype=float,
            ).ravel()
            / conservation_scales
        )
        vertical = (
            np.asarray(
                action["vertical_storage_per_ct"],
                dtype=float,
            ).ravel()
            / conservation_scales
        )
        total = (
            np.asarray(
                action["total_conservation_storage_per_ct"],
                dtype=float,
            ).ravel()
            / conservation_scales
        )
        return {
            "conserved": conserved,
            "vertical": vertical,
            "total": total,
        }

    if derivative_backend == "nested_matrix":
        component_action = nested_storage_rate_components_at_increment
    else:
        component_action = direct_storage_rate_components_at_increment

    derivatives, derivative_color_count = (
        _causal_five_field_colored_component_jacobians(
            component_action,
            np.zeros(n_reduced, dtype=float),
            storage_pattern,
            finite_difference_step=outer_step,
        )
    )
    if derivative_backend == "nested_matrix" and (
        not nested_storage_color_counts
        or any(
            count != storage_color_count
            for count in nested_storage_color_counts
        )
    ):
        raise RuntimeError(
            "nested storage coloring changed across tangent evaluations"
        )
    total = derivatives["conserved"] + derivatives["vertical"]
    component_defect = derivatives["total"] - total
    return {
        "storage_rate_derivative_scaled_matrix": total,
        "conserved_storage_rate_derivative_scaled_matrix": (
            derivatives["conserved"]
        ),
        "vertical_storage_rate_derivative_scaled_matrix": (
            derivatives["vertical"]
        ),
        "primitive_column_scales": primitive_scales,
        "conservation_row_scales": conservation_scales,
        "storage_matrix_difference_step": inner_step,
        "storage_rate_derivative_step": outer_step,
        "storage_difference_step": float(storage_difference_step),
        "storage_quadrature_order": int(storage_quadrature_order),
        "storage_directional_step": float(storage_directional_step),
        "conserved_difference_order": int(conserved_difference_order),
        "storage_component_colors": storage_color_count,
        "storage_rate_derivative_component_colors": (
            derivative_color_count
        ),
        "storage_rate_derivative_backend": derivative_backend,
        "storage_rate_derivative_uses_inner_storage_matrix": (
            derivative_backend == "nested_matrix"
        ),
        "storage_matrix_difference_step_applied": (
            inner_step if derivative_backend == "nested_matrix" else None
        ),
        "storage_rate_derivative_outer_component_evaluations": (
            2 * derivative_color_count
        ),
        "storage_rate_derivative_direct_action_evaluations": (
            2 * derivative_color_count
            if derivative_backend == "direct_action"
            else 0
        ),
        "storage_rate_derivative_nested_component_evaluations": (
            4 * derivative_color_count * storage_color_count
            if derivative_backend == "nested_matrix"
            else 0
        ),
        "storage_rate_derivative_nested_base_mapped_evaluations": (
            2 * derivative_color_count
            if derivative_backend == "nested_matrix"
            else 0
        ),
        "storage_rate_derivative_nested_mapped_evaluations": (
            2
            * derivative_color_count
            * (1 + 2 * storage_color_count)
            if derivative_backend == "nested_matrix"
            else 0
        ),
        "storage_rate_derivative_direct_mapped_evaluations": (
            2
            * int(conserved_difference_order)
            * derivative_color_count
            if derivative_backend == "direct_action"
            else 0
        ),
        "vertical_storage_rate_derivative_path_evaluations": (
            4 * derivative_color_count
        ),
        "maximum_scaled_component_reconstruction_defect": float(
            np.max(np.abs(component_defect))
        ),
        "storage_rate_derivative_source": (
            "independent_nested_colored_mapped_plus_vertical_rate_action"
            if derivative_backend == "nested_matrix"
            else (
                "independent_outer_colored_complete_storage_rate_action"
                if int(conserved_difference_order) == 2
                else (
                    "independent_outer_colored_complete_storage_rate_action_"
                    f"mapped_order_{int(conserved_difference_order)}"
                )
            )
        ),
    }


def causal_five_field_reduced_storage_rate_directional_derivative(
    context: CausalFiveFieldDAEContext,
    primitive_vector: np.ndarray,
    primitive_rate_per_s: np.ndarray,
    scaled_primitive_direction: np.ndarray,
    *,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    storage_rate_derivative_step: float,
    storage_difference_step: float,
    storage_quadrature_order: int = 4,
    storage_directional_step: float = 1.0e-3,
    conserved_difference_order: int = 2,
) -> dict:
    """Apply ``DM[direction, p_dot]`` without assembling its full matrix.

    This is the directional analogue of the ``direct_action`` backend above.
    It is primarily useful for step-convergence audits: changing the inner
    storage-action path can be tested in physically relevant directions
    without rebuilding every colored column.
    """

    context = context.validated()
    n_reduced = 5 * int(context.grid.centers.size)
    primitives = np.asarray(primitive_vector, dtype=float)
    physical_rate = np.asarray(primitive_rate_per_s, dtype=float).ravel()
    direction = np.asarray(scaled_primitive_direction, dtype=float).ravel()
    primitive_scales = np.asarray(primitive_column_scales, dtype=float)
    conservation_scales = np.asarray(conservation_row_scales, dtype=float)
    outer_step = float(storage_rate_derivative_step)
    if (
        primitives.shape != (n_reduced,)
        or physical_rate.shape != (n_reduced,)
        or direction.shape != (n_reduced,)
        or primitive_scales.shape != (n_reduced,)
        or conservation_scales.shape != (n_reduced,)
        or np.any(~np.isfinite(primitives))
        or np.any(~np.isfinite(physical_rate))
        or np.any(~np.isfinite(direction))
        or np.any(~np.isfinite(primitive_scales))
        or np.any(~np.isfinite(conservation_scales))
        or np.any(primitive_scales <= 0.0)
        or np.any(conservation_scales <= 0.0)
        or not np.isfinite(outer_step)
        or outer_step <= 0.0
    ):
        raise ValueError("storage-rate directional inputs are invalid")

    def action(sign: float) -> dict[str, np.ndarray]:
        perturbed = (
            primitives + sign * outer_step * primitive_scales * direction
        )
        result = _causal_five_field_reduced_storage_action_with_scale(
            context,
            perturbed,
            physical_rate,
            primitive_scales,
            storage_difference_step=storage_difference_step,
            storage_quadrature_order=storage_quadrature_order,
            storage_directional_step=storage_directional_step,
            conserved_difference_order=conserved_difference_order,
            include_conserved=True,
        )
        conserved = np.asarray(
            result["conserved_storage_per_ct"], dtype=float
        ).ravel() / conservation_scales
        vertical = np.asarray(
            result["vertical_storage_per_ct"], dtype=float
        ).ravel() / conservation_scales
        return {
            "conserved": conserved,
            "vertical": vertical,
            "total": conserved + vertical,
        }

    plus = action(1.0)
    minus = action(-1.0)
    denominator = 2.0 * outer_step
    conserved = (plus["conserved"] - minus["conserved"]) / denominator
    vertical = (plus["vertical"] - minus["vertical"]) / denominator
    return {
        "storage_rate_directional_derivative_scaled": conserved + vertical,
        "conserved_storage_rate_directional_derivative_scaled": conserved,
        "vertical_storage_rate_directional_derivative_scaled": vertical,
        "storage_rate_derivative_step": outer_step,
        "storage_difference_step": float(storage_difference_step),
        "storage_quadrature_order": int(storage_quadrature_order),
        "storage_directional_step": float(storage_directional_step),
        "conserved_difference_order": int(conserved_difference_order),
        "storage_action_evaluations": 2,
        "source": "centered_outer_complete_storage_rate_directional_action",
    }


def causal_five_field_assemble_evolving_tangent(
    descriptor_reduced_scaled_matrix: np.ndarray,
    stationary_reduced_scaled_jacobian: np.ndarray,
    storage_rate_derivative_scaled_matrix: np.ndarray,
) -> dict:
    """Assemble one evolving generator from independently audited blocks."""

    mass = np.asarray(descriptor_reduced_scaled_matrix, dtype=float)
    stationary = np.asarray(
        stationary_reduced_scaled_jacobian,
        dtype=float,
    )
    storage_rate = np.asarray(
        storage_rate_derivative_scaled_matrix,
        dtype=float,
    )
    if (
        mass.ndim != 2
        or mass.shape[0] != mass.shape[1]
        or stationary.shape != mass.shape
        or storage_rate.shape != mass.shape
        or np.any(~np.isfinite(mass))
        or np.any(~np.isfinite(stationary))
        or np.any(~np.isfinite(storage_rate))
    ):
        raise ValueError("evolving-tangent blocks are invalid")
    evolving = stationary + storage_rate
    generator = -np.linalg.solve(mass, evolving)
    factorization_defect = mass @ generator + evolving
    return {
        "descriptor_reduced_scaled_matrix": mass,
        "stationary_reduced_scaled_jacobian": stationary,
        "storage_rate_derivative_scaled_matrix": storage_rate,
        "evolving_reduced_scaled_jacobian": evolving,
        "evolving_scaled_generator_per_s": generator,
        "maximum_scaled_generator_factorization_defect": float(
            np.max(np.abs(factorization_defect))
        ),
        "dimensions": mass.shape,
    }


def causal_five_field_scaled_primitive_vector_field(
    context: CausalFiveFieldDAEContext,
    primitive_vector: np.ndarray,
    *,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    finite_difference_step: float = 2.0e-6,
    storage_quadrature_order: int = 4,
    storage_directional_step: float = 1.0e-3,
) -> dict:
    """Evaluate the implemented nonlinear scaled primitive vector field.

    The supplied scales remain fixed, so centered evaluations at neighboring
    primitive states define an independent JVP of the same scaled vector
    field.  This helper builds the coordinatewise mapped-storage matrix and
    stationary residual directly; it does not construct ``DM`` or an evolving
    tangent.
    """

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    n_reduced = 5 * n_cells
    primitives = np.asarray(primitive_vector, dtype=float)
    primitive_scales = np.asarray(
        primitive_column_scales,
        dtype=float,
    )
    conservation_scales = np.asarray(
        conservation_row_scales,
        dtype=float,
    )
    step = float(finite_difference_step)
    if (
        primitives.shape != (n_reduced,)
        or primitive_scales.shape != (n_reduced,)
        or conservation_scales.shape != (n_reduced,)
        or np.any(~np.isfinite(primitives))
        or np.any(~np.isfinite(primitive_scales))
        or np.any(~np.isfinite(conservation_scales))
        or np.any(primitive_scales <= 0.0)
        or np.any(conservation_scales <= 0.0)
        or not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("scaled primitive vector-field inputs are invalid")
    component_matrices, color_count = (
        _causal_five_field_reduced_storage_matrix_with_scale(
            context,
            primitives,
            primitive_scales,
            conservation_scales,
            finite_difference_step=step,
            storage_quadrature_order=storage_quadrature_order,
            storage_directional_step=storage_directional_step,
        )
    )
    stationary_residual = np.asarray(
        causal_five_field_reduced_stationary_residual(
            primitives,
            context,
        ),
        dtype=float,
    )
    scaled_stationary_residual = (
        stationary_residual / conservation_scales
    )
    mass = component_matrices["total"]
    scaled_rate = np.linalg.solve(
        mass,
        -scaled_stationary_residual,
    )
    return {
        "scaled_primitive_rate_per_s": scaled_rate.reshape(
            n_cells,
            5,
        ),
        "primitive_rate_per_s": (
            primitive_scales * scaled_rate
        ).reshape(n_cells, 5),
        "scaled_stationary_residual": (
            scaled_stationary_residual.reshape(n_cells, 5)
        ),
        "descriptor_reduced_scaled_matrix": mass,
        "conserved_descriptor_reduced_scaled_matrix": (
            component_matrices["conserved"]
        ),
        "vertical_descriptor_reduced_scaled_matrix": (
            component_matrices["vertical"]
        ),
        "primitive_column_scales": primitive_scales,
        "conservation_row_scales": conservation_scales,
        "finite_difference_step": step,
        "storage_component_colors": color_count,
        "storage_component_paired_evaluations": 2 * color_count,
        "mass_matrix_source": (
            "direct_gauss_mapped_vector_storage_one_form"
        ),
    }


def causal_five_field_evolving_tangent_matrices(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    *,
    primitive_rate_per_s: np.ndarray | None = None,
    reduced_descriptor: dict | None = None,
    finite_difference_step: float = 2.0e-6,
    descriptor_timestep_seconds: float = 1.0,
    storage_difference_step: float = 1.0e-4,
    storage_rate_derivative_step: float | None = None,
    storage_quadrature_order: int = 4,
    storage_directional_step: float = 1.0e-3,
) -> dict:
    """Return the primitive tangent at one evolving descriptor anchor.

    For the reduced nonlinear descriptor ``M(p) p_dot + S(p) = 0``, the
    evolving-anchor tangent is

    ``M delta_p_dot + (DS + DM[., p_dot]) delta_p = 0``.

    The historical frozen descriptor is preserved.  This helper adds the
    state-dependent storage term by differentiating the same coordinatewise
    colored storage matrix used by the nonlinear vector field.  The nested
    colored contraction avoids replacing the implemented mapped conserved
    Jacobian by one finite displacement along the full primitive rate.  The
    local responsive-height one-form retains its cheaper direct rate action.
    Both derivative components share each outer colored evaluation.  Returned
    matrices act on the fixed scaled primitive coordinates at the supplied
    anchor.  ``storage_rate_derivative_step`` controls that outer derivative
    independently of the inner mass-matrix finite difference and the
    directional height-storage action.
    """

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    n_reduced = 5 * n_cells
    step = float(finite_difference_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(
            "finite_difference_step must be positive and finite"
        )
    if storage_rate_derivative_step is None:
        raise ValueError(
            "storage_rate_derivative_step must be supplied explicitly"
        )
    rate_derivative_step = float(storage_rate_derivative_step)
    if (
        not np.isfinite(rate_derivative_step)
        or rate_derivative_step <= 0.0
    ):
        raise ValueError(
            "storage_rate_derivative_step must be positive and finite"
        )
    state = unpack_causal_five_field_state(vector, n_cells)
    base_primitives = np.asarray(state.primitives, dtype=float).ravel()
    evaluation = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    primitive_scales = np.asarray(
        scaling.column_scales[n_reduced : 2 * n_reduced],
        dtype=float,
    )
    conservation_scales = np.asarray(
        scaling.row_scales[:n_reduced],
        dtype=float,
    )
    if reduced_descriptor is None:
        frozen = causal_five_field_reduced_descriptor_matrices(
            context,
            vector,
            finite_difference_step=step,
            descriptor_timestep_seconds=descriptor_timestep_seconds,
        )
    else:
        frozen = reduced_descriptor
        if (
            tuple(frozen.get("dimensions", ())) != (
                n_reduced,
                n_reduced,
            )
            or not np.isclose(
                float(frozen.get("finite_difference_step", np.nan)),
                step,
                rtol=0.0,
                atol=0.0,
            )
            or not np.isclose(
                float(
                    frozen.get(
                        "descriptor_timestep_seconds",
                        np.nan,
                    )
                ),
                float(descriptor_timestep_seconds),
                rtol=0.0,
                atol=0.0,
            )
        ):
            raise ValueError(
                "precomputed reduced descriptor is incompatible"
            )
    frozen_mass = np.asarray(
        frozen["descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    stationary_jacobian = np.asarray(
        frozen["stationary_reduced_scaled_jacobian"],
        dtype=float,
    )
    if (
        frozen_mass.shape != (n_reduced, n_reduced)
        or stationary_jacobian.shape != (n_reduced, n_reduced)
        or np.any(~np.isfinite(frozen_mass))
        or np.any(~np.isfinite(stationary_jacobian))
        or not np.allclose(
            np.asarray(frozen["primitive_column_scales"], dtype=float),
            primitive_scales,
            rtol=2.0e-14,
            atol=0.0,
        )
        or not np.allclose(
            np.asarray(frozen["conservation_row_scales"], dtype=float),
            conservation_scales,
            rtol=2.0e-14,
            atol=0.0,
        )
    ):
        raise ValueError(
            "precomputed reduced descriptor scaling or matrices differ"
        )
    storage_blocks = causal_five_field_reduced_storage_matrices(
        context,
        base_primitives,
        primitive_column_scales=primitive_scales,
        conservation_row_scales=conservation_scales,
        finite_difference_step=step,
        storage_quadrature_order=storage_quadrature_order,
        storage_directional_step=storage_directional_step,
    )
    mass = storage_blocks["descriptor_reduced_scaled_matrix"]
    conserved_mass = storage_blocks[
        "conserved_descriptor_reduced_scaled_matrix"
    ]
    vertical_mass = storage_blocks[
        "vertical_descriptor_reduced_scaled_matrix"
    ]
    storage_color_count = int(storage_blocks["storage_component_colors"])
    stationary_residual = causal_five_field_reduced_stationary_residual(
        base_primitives,
        context,
    )
    scaled_stationary_residual = (
        np.asarray(stationary_residual, dtype=float)
        / conservation_scales
    )

    if primitive_rate_per_s is None:
        scaled_rate = np.linalg.solve(
            mass,
            -scaled_stationary_residual,
        )
        physical_rate = primitive_scales * scaled_rate
        rate_source = "descriptor_balance"
    else:
        supplied = np.asarray(primitive_rate_per_s, dtype=float)
        if supplied.shape == (n_cells, 5):
            supplied = supplied.ravel()
        if (
            supplied.shape != (n_reduced,)
            or np.any(~np.isfinite(supplied))
        ):
            raise ValueError(
                "primitive_rate_per_s has the wrong shape or value"
            )
        physical_rate = supplied
        scaled_rate = physical_rate / primitive_scales
        rate_source = "supplied"

    storage_rate_blocks = (
        causal_five_field_reduced_storage_rate_derivatives(
            context,
            base_primitives,
            physical_rate,
            primitive_column_scales=primitive_scales,
            conservation_row_scales=conservation_scales,
            storage_matrix_difference_step=step,
            storage_rate_derivative_step=rate_derivative_step,
            storage_difference_step=storage_difference_step,
            storage_quadrature_order=storage_quadrature_order,
            storage_directional_step=storage_directional_step,
        )
    )
    conserved_storage_rate_derivative = storage_rate_blocks[
        "conserved_storage_rate_derivative_scaled_matrix"
    ]
    vertical_storage_rate_derivative = storage_rate_blocks[
        "vertical_storage_rate_derivative_scaled_matrix"
    ]
    storage_rate_derivative = storage_rate_blocks[
        "storage_rate_derivative_scaled_matrix"
    ]
    derivative_color_count = int(
        storage_rate_blocks[
            "storage_rate_derivative_component_colors"
        ]
    )
    assembled = causal_five_field_assemble_evolving_tangent(
        mass,
        stationary_jacobian,
        storage_rate_derivative,
    )
    evolving_jacobian = assembled["evolving_reduced_scaled_jacobian"]
    generator = assembled["evolving_scaled_generator_per_s"]
    base_storage_action = (
        _causal_five_field_reduced_storage_action_with_scale(
            context,
            base_primitives,
            physical_rate,
            primitive_scales,
            storage_difference_step=storage_difference_step,
            storage_quadrature_order=storage_quadrature_order,
            storage_directional_step=storage_directional_step,
        )
    )
    scaled_direct_storage = (
        np.asarray(
            base_storage_action[
                "total_conservation_storage_per_ct"
            ],
            dtype=float,
        ).ravel()
        / conservation_scales
    )
    scaled_matrix_storage = mass @ scaled_rate
    scaled_frozen_matrix_storage = frozen_mass @ scaled_rate
    storage_scale = max(
        float(np.max(np.abs(scaled_direct_storage))),
        float(np.max(np.abs(scaled_matrix_storage))),
        np.finfo(float).tiny,
    )
    frozen_storage_scale = max(
        storage_scale,
        float(np.max(np.abs(scaled_frozen_matrix_storage))),
    )
    direct_off_cell = np.array(mass, copy=True)
    frozen_off_cell = np.array(frozen_mass, copy=True)
    for cell in range(n_cells):
        local = slice(5 * cell, 5 * (cell + 1))
        direct_off_cell[local, local] = 0.0
        frozen_off_cell[local, local] = 0.0
    mass_difference = frozen_mass - mass
    mass_difference_index = np.unravel_index(
        int(np.argmax(np.abs(mass_difference))),
        mass_difference.shape,
    )
    generator_defect = mass @ generator + evolving_jacobian
    descriptor_component_defect = mass - (
        conserved_mass + vertical_mass
    )
    storage_rate_component_defect = storage_rate_derivative - (
        conserved_storage_rate_derivative
        + vertical_storage_rate_derivative
    )
    return {
        "descriptor_reduced_scaled_matrix": mass,
        "conserved_descriptor_reduced_scaled_matrix": conserved_mass,
        "vertical_descriptor_reduced_scaled_matrix": vertical_mass,
        "frozen_descriptor_reduced_scaled_matrix": frozen_mass,
        "stationary_reduced_scaled_jacobian": stationary_jacobian,
        "storage_rate_derivative_scaled_matrix": (
            storage_rate_derivative
        ),
        "conserved_storage_rate_derivative_scaled_matrix": (
            conserved_storage_rate_derivative
        ),
        "vertical_storage_rate_derivative_scaled_matrix": (
            vertical_storage_rate_derivative
        ),
        "evolving_reduced_scaled_jacobian": evolving_jacobian,
        "evolving_scaled_generator_per_s": generator,
        "primitive_rate_per_s": physical_rate.reshape(n_cells, 5),
        "scaled_primitive_rate_per_s": scaled_rate.reshape(n_cells, 5),
        "rate_source": rate_source,
        "primitive_column_scales": primitive_scales,
        "conservation_row_scales": conservation_scales,
        "base_storage_action": base_storage_action,
        "maximum_relative_storage_action_defect": float(
            np.max(
                np.abs(
                    scaled_direct_storage - scaled_matrix_storage
                )
            )
            / storage_scale
        ),
        "maximum_relative_frozen_storage_action_defect": float(
            np.max(
                np.abs(
                    scaled_direct_storage
                    - scaled_frozen_matrix_storage
                )
            )
            / frozen_storage_scale
        ),
        "maximum_relative_storage_matrix_change": float(
            np.max(np.abs(mass - frozen_mass))
            / max(
                float(np.max(np.abs(mass))),
                float(np.max(np.abs(frozen_mass))),
                np.finfo(float).tiny,
            )
        ),
        "maximum_absolute_storage_matrix_difference": float(
            np.max(np.abs(mass_difference))
        ),
        "maximum_storage_matrix_difference_row": int(
            mass_difference_index[0]
        ),
        "maximum_storage_matrix_difference_column": int(
            mass_difference_index[1]
        ),
        "maximum_absolute_direct_off_cell_storage_entry": float(
            np.max(np.abs(direct_off_cell))
        ),
        "maximum_absolute_frozen_off_cell_storage_entry": float(
            np.max(np.abs(frozen_off_cell))
        ),
        "direct_off_cell_storage_nonzero_count": int(
            np.count_nonzero(direct_off_cell)
        ),
        "maximum_scaled_generator_factorization_defect": float(
            np.max(np.abs(generator_defect))
        ),
        "maximum_scaled_descriptor_component_reconstruction_defect": float(
            np.max(np.abs(descriptor_component_defect))
        ),
        "maximum_scaled_storage_rate_component_reconstruction_defect": float(
            np.max(np.abs(storage_rate_component_defect))
        ),
        "dimensions": (n_reduced, n_reduced),
        "finite_difference_step": step,
        "storage_difference_step": float(storage_difference_step),
        "storage_rate_derivative_step": rate_derivative_step,
        "storage_quadrature_order": int(storage_quadrature_order),
        "storage_directional_step": float(storage_directional_step),
        "storage_component_colors": storage_color_count,
        "storage_rate_derivative_component_colors": (
            derivative_color_count
        ),
        "storage_rate_derivative_inner_component_colors": (
            storage_color_count
        ),
        "storage_rate_derivative_outer_component_evaluations": (
            2 * derivative_color_count
        ),
        "storage_rate_derivative_nested_component_evaluations": (
            4 * derivative_color_count * storage_color_count
        ),
        "storage_rate_derivative_nested_base_mapped_evaluations": (
            2 * derivative_color_count
        ),
        "storage_rate_derivative_nested_mapped_evaluations": (
            2
            * derivative_color_count
            * (1 + 2 * storage_color_count)
        ),
        "vertical_storage_rate_derivative_outer_action_evaluations": (
            2 * derivative_color_count
        ),
        "vertical_storage_rate_derivative_path_evaluations": (
            4 * derivative_color_count
        ),
        "storage_rate_derivative_source": (
            "nested_colored_conserved_matrix_plus_vertical_rate_action"
        ),
        "storage_local_block_size": None,
        "mass_matrix_source": (
            "direct_gauss_mapped_vector_storage_one_form"
        ),
        "frozen_descriptor": frozen,
    }


def causal_five_field_consistent_tangent_decomposition(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    *,
    finite_difference_step: float = 2.0e-6,
    rank_relative_threshold: float | None = None,
    linear_solver: str = "auto",
) -> dict:
    """Solve the DAE-consistent tangent for each signed stationary forcing."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    (
        state,
        evaluation,
        scaling,
        stationary_sparse,
        backward_euler_sparse,
    ) = _causal_five_field_scaled_stationary_and_be_jacobians(
        context,
        vector,
        finite_difference_step=finite_difference_step,
        descriptor_timestep_seconds=1.0,
    )

    solver = str(linear_solver)
    if solver not in ("auto", "dense", "sparse"):
        raise ValueError("consistent tangent linear solver is unsupported")
    if solver == "auto":
        solver = "sparse" if n_cells > 64 else "dense"

    n_differential = 5 * n_cells
    consistency_sparse = vstack(
        (
            (backward_euler_sparse - stationary_sparse)[:n_differential],
            stationary_sparse[n_differential:],
        ),
        format="csr",
    )
    if rank_relative_threshold is None:
        consistency_rank = None
        consistency_condition = None
    else:
        consistency_matrix = consistency_sparse.toarray()
        threshold = float(rank_relative_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "rank_relative_threshold must be positive and finite"
            )
        singular_values = np.linalg.svd(
            consistency_matrix,
            compute_uv=False,
        )
        consistency_rank = int(
            np.count_nonzero(
                singular_values
                > threshold * float(singular_values[0])
            )
        )
        consistency_condition = float(
            singular_values[0] / singular_values[-1]
        )
    scaled_stationary_residual = (
        np.asarray(evaluation.residual, dtype=float)
        / np.asarray(scaling.row_scales, dtype=float)
    )
    full_right_hand_side = np.concatenate(
        (
            -scaled_stationary_residual[:n_differential],
            np.zeros(
                stationary_sparse.shape[0] - n_differential,
                dtype=float,
            ),
        )
    )
    signed_terms = causal_five_field_residual_terms(
        context,
        vector,
        evaluation,
    )
    forcing_terms = {
        name: values
        for name, values in signed_terms.items()
        if name
        not in (
            "temporal_conserved_storage",
            "temporal_vertical_storage",
        )
    }
    term_names = tuple(forcing_terms)
    row_scale = np.asarray(
        scaling.row_scales[:n_differential],
        dtype=float,
    ).reshape(n_cells, 5)
    component_right_hand_sides = np.zeros(
        (consistency_sparse.shape[0], len(term_names)),
        dtype=float,
    )
    for index, name in enumerate(term_names):
        component_right_hand_sides[:n_differential, index] = (
            -np.asarray(forcing_terms[name], dtype=float).ravel()
            / row_scale.ravel()
        )
    all_right_hand_sides = np.column_stack(
        (full_right_hand_side, component_right_hand_sides)
    )
    if solver == "dense":
        consistency_matrix = consistency_sparse.toarray()
        all_scaled_tangents = np.linalg.solve(
            consistency_matrix,
            all_right_hand_sides,
        )
    else:
        row_maximum = np.asarray(
            np.abs(consistency_sparse).max(axis=1).toarray(),
            dtype=float,
        ).ravel()
        if np.any(row_maximum <= np.finfo(float).tiny):
            raise np.linalg.LinAlgError(
                "consistent tangent matrix has a zero row"
            )
        row_scale = 1.0 / row_maximum
        row_scaled = diags(row_scale) @ consistency_sparse
        column_maximum = np.asarray(
            np.abs(row_scaled).max(axis=0).toarray(),
            dtype=float,
        ).ravel()
        if np.any(column_maximum <= np.finfo(float).tiny):
            raise np.linalg.LinAlgError(
                "consistent tangent matrix has a zero column"
            )
        column_scale = 1.0 / column_maximum
        balanced = (
            diags(row_scale)
            @ consistency_sparse
            @ diags(column_scale)
        ).tocsc()
        factorization = splu(balanced)
        balanced_tangents = factorization.solve(
            row_scale[:, None] * all_right_hand_sides
        )
        all_scaled_tangents = column_scale[:, None] * balanced_tangents
    scaled_full_tangent = all_scaled_tangents[:, 0]
    scaled_component_tangents = all_scaled_tangents[:, 1:]
    column_scale = np.asarray(scaling.column_scales, dtype=float)
    full_physical_tangent = column_scale * scaled_full_tangent
    component_physical_tangents = (
        column_scale[:, None] * scaled_component_tangents
    )
    primitive_slice = slice(n_differential, 2 * n_differential)
    full_primitive = full_physical_tangent[primitive_slice].reshape(
        n_cells,
        5,
    )
    full_conserved = full_physical_tangent[:n_differential].reshape(
        n_cells,
        5,
    )
    full_log_h = causal_five_field_log_h_over_r_tangent(
        context,
        vector,
        full_primitive,
    )
    components = {}
    for index, name in enumerate(term_names):
        physical = component_physical_tangents[:, index]
        primitive = physical[primitive_slice].reshape(n_cells, 5)
        conserved = physical[:n_differential].reshape(n_cells, 5)
        components[name] = {
            "primitive_tangent_per_s": primitive,
            "conserved_tangent_per_s": conserved,
            "log_h_over_r_tangent_per_s": (
                causal_five_field_log_h_over_r_tangent(
                    context,
                    vector,
                    primitive,
                )
            ),
        }
    component_sum = np.sum(component_physical_tangents, axis=1)
    tangent_scale = np.maximum(
        np.abs(full_physical_tangent),
        np.sum(np.abs(component_physical_tangents), axis=1),
    )
    tangent_scale = np.maximum(tangent_scale, 1.0e-300)
    consistency_defect = (
        consistency_sparse @ scaled_full_tangent
        - full_right_hand_side
    )
    term_defect = causal_five_field_term_reconstruction_defect(
        evaluation,
        signed_terms,
    )
    return {
        "radius_rg": (
            np.asarray(context.grid.centers, dtype=float)
            / context.grid.gravitational_radius
        ),
        "cell_measures": np.asarray(
            context.grid.cell_measures,
            dtype=float,
        ),
        "full": {
            "physical_tangent_per_s": full_physical_tangent,
            "primitive_tangent_per_s": full_primitive,
            "conserved_tangent_per_s": full_conserved,
            "log_h_over_r_tangent_per_s": full_log_h,
        },
        "components": components,
        "term_names": term_names,
        "consistency_dimensions": consistency_sparse.shape,
        "consistency_numerical_rank": consistency_rank,
        "consistency_condition_estimate": consistency_condition,
        "linear_solver": solver,
        "consistency_nonzeros": int(consistency_sparse.nnz),
        "maximum_scaled_consistency_defect": float(
            np.max(np.abs(consistency_defect))
        ),
        "maximum_residual_reconstruction_relative_defect": (
            term_defect["maximum_relative_defect"]
        ),
        "maximum_tangent_reconstruction_relative_defect": float(
            np.max(
                np.abs(component_sum - full_physical_tangent)
                / tangent_scale
            )
        ),
        "outer_boundary_choked": bool(
            evaluation.outer_boundary_choked
        ),
        "outer_incoming_characteristics": int(
            evaluation.outer_incoming_characteristics
        ),
    }
