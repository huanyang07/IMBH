"""Nested-grid diagnostics for the causal five-field finite-volume system."""

from __future__ import annotations

import numpy as np

from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    CausalFiveFieldDAEEvaluation,
    causal_five_field_colored_central_jacobian,
    causal_five_field_cell_states,
    causal_five_field_dae_jacobian_sparsity,
    causal_five_field_dae_scaling,
    evaluate_causal_five_field_dae,
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
        "central_face_transport": central[1:] - central[:-1],
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


def causal_five_field_consistent_tangent_decomposition(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    *,
    finite_difference_step: float = 2.0e-6,
    rank_relative_threshold: float | None = None,
) -> dict:
    """Solve the DAE-consistent tangent for each signed stationary forcing."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(vector, n_cells)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    zero = np.zeros_like(np.asarray(vector, dtype=float))
    pattern = causal_five_field_dae_jacobian_sparsity(
        n_cells,
        spatial_reconstruction=context.spatial_reconstruction,
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
                timestep_seconds=1.0,
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
    ).toarray()
    backward_euler = causal_five_field_colored_central_jacobian(
        lambda increment: scaled_residual(
            increment,
            backward_euler=True,
        ),
        zero,
        pattern,
        finite_difference_step=finite_difference_step,
    ).toarray()
    n_differential = 5 * n_cells
    consistency_matrix = np.vstack(
        (
            (backward_euler - stationary)[:n_differential],
            stationary[n_differential:],
        )
    )
    if rank_relative_threshold is None:
        consistency_rank = None
        consistency_condition = None
    else:
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
            np.zeros(stationary.shape[0] - n_differential, dtype=float),
        )
    )
    scaled_full_tangent = np.linalg.solve(
        consistency_matrix,
        full_right_hand_side,
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
        (consistency_matrix.shape[0], len(term_names)),
        dtype=float,
    )
    for index, name in enumerate(term_names):
        component_right_hand_sides[:n_differential, index] = (
            -np.asarray(forcing_terms[name], dtype=float).ravel()
            / row_scale.ravel()
        )
    scaled_component_tangents = np.linalg.solve(
        consistency_matrix,
        component_right_hand_sides,
    )
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
        consistency_matrix @ scaled_full_tangent
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
        "consistency_dimensions": consistency_matrix.shape,
        "consistency_numerical_rank": consistency_rank,
        "consistency_condition_estimate": consistency_condition,
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
