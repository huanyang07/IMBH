"""Audit-only shear-energy ledgers for the frozen causal inner generator.

The production DAE and its numerical flux are not changed here.  This module
builds three independent pieces needed by WP10c9c0b:

* a positive, basis-invariant shear energy and an energy-orthogonal
  one-family/complement partition;
* an exact physical decomposition of the cached evolving primitive generator
  into transport, source, and descriptor-rate blocks;
* instantaneous and integrated quadratic-energy ledgers for an unchanged
  frozen trajectory.

The component generator is reconstructed in the same fixed scaled primitive
coordinates as the cached tangent.  Any finite-difference or assembly
difference is retained explicitly as ``residual_unattributed`` so every audit
ledger sums exactly to the supplied full generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import splu

from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    causal_five_field_dae_jacobian_color_groups,
    causal_five_field_dae_scaling,
    causal_five_field_face_flux_decomposition,
    causal_five_field_reduced_stationary_residual,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    pack_causal_five_field_state,
)
from .causal_inner_shear_root_cause import (
    causal_five_field_shear_invariant_subspace,
)
from .causal_inner_spatial_audit import (
    causal_five_field_reduced_stationary_jacobian,
    causal_five_field_reduced_storage_matrices,
    causal_five_field_reduced_storage_rate_derivatives,
)


CAUSAL_SHEAR_ENERGY_FAMILIES = ("inward_shear", "outward_shear")
_N_FIELDS = 5


@dataclass(frozen=True)
class CausalShearEnergyProjectors:
    """Cellwise physical shear metrics and orthogonal family projectors."""

    primitive_shear_projectors: np.ndarray
    primitive_energy_grams: np.ndarray
    primitive_family_projectors: np.ndarray
    primitive_family_complement_projectors: np.ndarray
    primitive_family_energy_grams: np.ndarray
    primitive_family_complement_energy_grams: np.ndarray
    minimum_positive_energy_eigenvalue: float
    maximum_shear_projector_defect: float
    maximum_family_projector_defect: float
    maximum_partition_defect: float
    maximum_energy_self_adjoint_defect: float
    maximum_energy_partition_defect: float


@dataclass(frozen=True)
class CausalGeneratorBlockDecomposition:
    """Physical blocks whose exact sum is one supplied evolving generator."""

    full_generator_per_s: np.ndarray
    generator_blocks_per_s: dict[str, np.ndarray]
    descriptor_matrix: np.ndarray
    scaled_primitive_rate_per_s: np.ndarray
    physical_primitive_rate_per_s: np.ndarray
    component_names: tuple[str, ...]
    reduced_pattern_colors: int
    maximum_base_residual_reconstruction_defect: float
    maximum_stationary_jacobian_reconstruction_defect: float
    maximum_generator_reconstruction_defect_before_remainder: float
    maximum_generator_reconstruction_defect_after_remainder: float
    maximum_mass_solve_relative_defect: float
    residual_unattributed_relative_frobenius_norm: float


@dataclass(frozen=True)
class CausalShearEnergyLedger:
    """Exact quadratic-energy rates for one unchanged state history."""

    family: str
    times_seconds: np.ndarray
    total_energy: np.ndarray
    selected_energy: np.ndarray
    complement_energy: np.ndarray
    total_energy_rate_per_s: np.ndarray
    selected_energy_rate_per_s: np.ndarray
    complement_energy_rate_per_s: np.ndarray
    total_rate_by_block_per_s: dict[str, np.ndarray]
    selected_rate_by_block_per_s: dict[str, np.ndarray]
    complement_rate_by_block_per_s: dict[str, np.ndarray]
    selected_rate_by_source_partition_per_s: dict[str, np.ndarray]
    total_rate_by_source_partition_per_s: dict[str, np.ndarray]
    preserving_total_rate_per_s: np.ndarray
    transfer_total_rate_per_s: np.ndarray
    preserving_selected_rate_per_s: np.ndarray
    transfer_selected_rate_per_s: np.ndarray
    cumulative_total_rate_integral: np.ndarray
    cumulative_selected_rate_integral: np.ndarray
    cumulative_complement_rate_integral: np.ndarray
    maximum_instantaneous_energy_partition_defect: float
    maximum_instantaneous_block_ledger_defect: float
    maximum_instantaneous_source_partition_defect: float
    maximum_integrated_total_ledger_defect: float
    maximum_integrated_selected_ledger_defect: float
    maximum_integrated_complement_ledger_defect: float


def _relative_maximum_defect(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_values = np.asarray(first)
    second_values = np.asarray(second)
    scale = max(
        float(np.max(np.abs(first_values))),
        float(np.max(np.abs(second_values))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(first_values - second_values)) / scale)


def _family_index(family: str) -> int:
    try:
        return CAUSAL_SHEAR_ENERGY_FAMILIES.index(str(family))
    except ValueError as exc:
        raise ValueError("unknown causal shear-energy family") from exc


def causal_five_field_shear_energy_projectors(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
) -> CausalShearEnergyProjectors:
    """Build an energy-orthogonal family/complement split in every cell.

    The selected one-dimensional subspace is the complete-coordinate
    characteristic branch.  Its complement is the orthogonal complement
    inside the two-shear subspace under the positive local-rest shear energy.
    This preserves the selected physical direction while removing the large,
    basis-normalization-dependent cross term found in WP10c9c0.
    """

    context = context.validated()
    charts = np.asarray(primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    if (
        charts.shape != (n_cells, _N_FIELDS)
        or np.any(~np.isfinite(charts))
    ):
        raise ValueError("shear-energy projector charts are invalid")

    shear_projectors = np.empty((n_cells, _N_FIELDS, _N_FIELDS))
    energy_grams = np.empty_like(shear_projectors)
    family_projectors = np.empty(
        (2, n_cells, _N_FIELDS, _N_FIELDS),
        dtype=float,
    )
    complement_projectors = np.empty_like(family_projectors)
    family_grams = np.empty_like(family_projectors)
    complement_grams = np.empty_like(family_projectors)
    minimum_positive_energy = float("inf")
    maximum_shear_projector = 0.0
    maximum_family_projector = 0.0
    maximum_partition = 0.0
    maximum_self_adjoint = 0.0
    maximum_energy_partition = 0.0

    for cell, (radius, chart) in enumerate(
        zip(context.grid.centers, charts, strict=True)
    ):
        shear = causal_five_field_shear_invariant_subspace(
            context,
            float(radius),
            chart,
        )
        right = np.asarray(
            shear.primitive_right_eigenvectors,
            dtype=float,
        )
        left = np.asarray(
            shear.primitive_left_eigenvectors,
            dtype=float,
        )
        coefficient_gram = np.asarray(
            shear.coordinate_energy_gram,
            dtype=float,
        )
        projector = right @ left
        primitive_gram = left.T @ coefficient_gram @ left
        primitive_gram = 0.5 * (primitive_gram + primitive_gram.T)
        shear_projectors[cell] = projector
        energy_grams[cell] = primitive_gram
        positive = np.linalg.eigvalsh(coefficient_gram)
        minimum_positive_energy = min(
            minimum_positive_energy,
            float(np.min(positive)),
        )
        maximum_shear_projector = max(
            maximum_shear_projector,
            float(np.max(np.abs(projector @ projector - projector))),
        )

        for family_index in range(2):
            axis = np.zeros(2, dtype=float)
            axis[family_index] = 1.0
            denominator = float(
                axis @ coefficient_gram @ axis
            )
            if (
                not np.isfinite(denominator)
                or denominator <= np.finfo(float).tiny
            ):
                raise RuntimeError(
                    "causal shear family has non-positive energy"
                )
            coefficient_projector = np.outer(
                axis,
                axis @ coefficient_gram / denominator,
            )
            family_projector = (
                right @ coefficient_projector @ left
            )
            complement_projector = projector - family_projector
            family_gram = (
                family_projector.T
                @ primitive_gram
                @ family_projector
            )
            complement_gram = (
                complement_projector.T
                @ primitive_gram
                @ complement_projector
            )
            family_gram = 0.5 * (family_gram + family_gram.T)
            complement_gram = 0.5 * (
                complement_gram + complement_gram.T
            )
            family_projectors[family_index, cell] = family_projector
            complement_projectors[
                family_index,
                cell,
            ] = complement_projector
            family_grams[family_index, cell] = family_gram
            complement_grams[family_index, cell] = complement_gram
            maximum_family_projector = max(
                maximum_family_projector,
                float(
                    np.max(
                        np.abs(
                            family_projector @ family_projector
                            - family_projector
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            complement_projector
                            @ complement_projector
                            - complement_projector
                        )
                    )
                ),
            )
            maximum_partition = max(
                maximum_partition,
                float(
                    np.max(
                        np.abs(
                            family_projector
                            + complement_projector
                            - projector
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            family_projector @ complement_projector
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            complement_projector @ family_projector
                        )
                    )
                ),
            )
            maximum_self_adjoint = max(
                maximum_self_adjoint,
                _relative_maximum_defect(
                    family_projector.T @ primitive_gram,
                    primitive_gram @ family_projector,
                ),
                _relative_maximum_defect(
                    complement_projector.T @ primitive_gram,
                    primitive_gram @ complement_projector,
                ),
            )
            maximum_energy_partition = max(
                maximum_energy_partition,
                _relative_maximum_defect(
                    primitive_gram,
                    family_gram + complement_gram,
                ),
            )

    return CausalShearEnergyProjectors(
        primitive_shear_projectors=shear_projectors,
        primitive_energy_grams=energy_grams,
        primitive_family_projectors=family_projectors,
        primitive_family_complement_projectors=(
            complement_projectors
        ),
        primitive_family_energy_grams=family_grams,
        primitive_family_complement_energy_grams=complement_grams,
        minimum_positive_energy_eigenvalue=minimum_positive_energy,
        maximum_shear_projector_defect=maximum_shear_projector,
        maximum_family_projector_defect=maximum_family_projector,
        maximum_partition_defect=maximum_partition,
        maximum_energy_self_adjoint_defect=maximum_self_adjoint,
        maximum_energy_partition_defect=maximum_energy_partition,
    )


def causal_five_field_stationary_residual_components(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return non-overlapping physical pieces of the reduced residual."""

    context = context.validated()
    charts = np.asarray(primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    if (
        charts.shape != (n_cells, _N_FIELDS)
        or np.any(~np.isfinite(charts))
    ):
        raise ValueError("stationary component charts are invalid")
    state = causal_five_field_state_from_primitives(context, charts)
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    decomposition = causal_five_field_face_flux_decomposition(
        context,
        vector,
    )
    face_shape = (n_cells + 1, _N_FIELDS)
    component_faces = {
        "transport_inner_boundary": np.zeros(face_shape, dtype=float),
        "transport_central_perfect": np.zeros(face_shape, dtype=float),
        "transport_central_stress": np.zeros(face_shape, dtype=float),
        "transport_rusanov": np.zeros(face_shape, dtype=float),
        "transport_outer_boundary": np.zeros(face_shape, dtype=float),
    }
    component_faces["transport_inner_boundary"][0] = (
        evaluation.numerical_weighted_face_fluxes_over_c[0]
    )
    component_faces["transport_central_perfect"][1:-1] = (
        decomposition.central_perfect_weighted_face_fluxes_over_c
    )
    component_faces["transport_central_stress"][1:-1] = (
        decomposition.central_stress_weighted_face_fluxes_over_c
    )
    component_faces["transport_rusanov"][1:-1] = (
        decomposition.rusanov_weighted_face_fluxes_over_c
    )
    component_faces["transport_outer_boundary"][-1] = (
        evaluation.numerical_weighted_face_fluxes_over_c[-1]
    )
    result = {
        name: values[1:] - values[:-1]
        for name, values in component_faces.items()
    }
    for name, values in sorted(
        evaluation.integrated_source_components_per_ct.items()
    ):
        result[f"source_{name}"] = -np.asarray(values, dtype=float)
    return result


def _reduced_component_pattern(
    n_cells: int,
    *,
    stencil_radius: int,
) -> csr_matrix:
    n_cells = int(n_cells)
    radius = int(stencil_radius)
    if n_cells < 3 or radius < 1:
        raise ValueError("reduced component pattern inputs are invalid")
    n_reduced = _N_FIELDS * n_cells
    pattern = lil_matrix((n_reduced, n_reduced), dtype=np.int8)
    for output_cell in range(n_cells):
        first = max(0, output_cell - radius)
        last = min(n_cells, output_cell + radius + 1)
        rows = slice(
            _N_FIELDS * output_cell,
            _N_FIELDS * (output_cell + 1),
        )
        columns = slice(_N_FIELDS * first, _N_FIELDS * last)
        pattern[rows, columns] = 1
    return pattern.tocsr()


def _colored_stationary_component_jacobians(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    *,
    finite_difference_step: float,
    stencil_radius: int,
) -> tuple[dict[str, csr_matrix], int]:
    charts = np.asarray(primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    n_reduced = _N_FIELDS * n_cells
    column_scales = np.asarray(
        primitive_column_scales,
        dtype=float,
    ).reshape(n_reduced)
    row_scales = np.asarray(
        conservation_row_scales,
        dtype=float,
    ).reshape(n_reduced)
    step = float(finite_difference_step)
    if (
        charts.shape != (n_cells, _N_FIELDS)
        or np.any(~np.isfinite(charts))
        or np.any(~np.isfinite(column_scales))
        or np.any(column_scales <= 0.0)
        or np.any(~np.isfinite(row_scales))
        or np.any(row_scales <= 0.0)
        or not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("component-Jacobian inputs are invalid")
    pattern = _reduced_component_pattern(
        n_cells,
        stencil_radius=stencil_radius,
    )
    declared = pattern.tocsc()
    groups = causal_five_field_dae_jacobian_color_groups(pattern)
    matrices: dict[str, lil_matrix] | None = None
    base = charts.ravel()
    for group in groups:
        increment = np.zeros(n_reduced, dtype=float)
        increment[group] = step
        plus = causal_five_field_stationary_residual_components(
            context,
            (base + column_scales * increment).reshape(
                n_cells,
                _N_FIELDS,
            ),
        )
        minus = causal_five_field_stationary_residual_components(
            context,
            (base - column_scales * increment).reshape(
                n_cells,
                _N_FIELDS,
            ),
        )
        if matrices is None:
            if tuple(plus) != tuple(minus):
                raise RuntimeError(
                    "stationary component schemas differ at first color"
                )
            matrices = {
                name: lil_matrix((n_reduced, n_reduced), dtype=float)
                for name in plus
            }
        elif (
            tuple(plus) != tuple(matrices)
            or tuple(minus) != tuple(matrices)
        ):
            raise RuntimeError(
                "stationary component schema changed under perturbation"
            )
        differences = {
            name: (
                (
                    np.asarray(plus[name], dtype=float)
                    - np.asarray(minus[name], dtype=float)
                ).ravel()
                / row_scales
                / (2.0 * step)
            )
            for name in plus
        }
        for column in group:
            start = declared.indptr[column]
            stop = declared.indptr[column + 1]
            rows = declared.indices[start:stop]
            for name, difference in differences.items():
                matrices[name][rows, column] = difference[rows, None]
    if matrices is None:
        raise RuntimeError("stationary component coloring is empty")
    return {
        name: matrix.tocsr()
        for name, matrix in matrices.items()
    }, len(groups)


def _sparse_mass_solve(
    mass: np.ndarray,
    right_hand_side: np.ndarray,
) -> tuple[np.ndarray, float]:
    matrix = csr_matrix(np.asarray(mass, dtype=float)).tocsc()
    right = np.asarray(right_hand_side, dtype=float)
    if (
        matrix.shape[0] != matrix.shape[1]
        or right.shape[0] != matrix.shape[0]
        or np.any(~np.isfinite(matrix.data))
        or np.any(~np.isfinite(right))
    ):
        raise ValueError("mass solve inputs are invalid")
    factorization = splu(matrix)
    solution = np.asarray(factorization.solve(right), dtype=float)
    residual = matrix @ solution - right
    relative = float(
        np.max(np.abs(residual))
        / max(float(np.max(np.abs(right))), np.finfo(float).tiny)
    )
    return solution, relative


def causal_five_field_generator_block_decomposition(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    *,
    primitive_column_scales: np.ndarray,
    full_generator_per_s: np.ndarray,
    primitive_rate_per_s: np.ndarray | None = None,
    finite_difference_step: float = 2.0e-6,
    storage_rate_derivative_step: float = 2.0e-6,
    storage_difference_step: float = 1.0e-4,
    storage_quadrature_order: int = 4,
    storage_directional_step: float = 1.0e-3,
    stencil_radius: int = 4,
) -> CausalGeneratorBlockDecomposition:
    """Decompose one cached evolving generator into exact physical blocks.

    ``primitive_rate_per_s`` may be supplied when the generator comes from a
    frozen legacy cache.  The storage-rate derivative is a nested numerical
    derivative, so reproducing that generator requires the exact physical
    base rate used when the cache was constructed rather than a newly solved
    rate that differs only at finite-difference roundoff.
    """

    context = context.validated()
    charts = np.asarray(primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    n_reduced = _N_FIELDS * n_cells
    scales = np.asarray(primitive_column_scales, dtype=float).reshape(
        n_reduced
    )
    full = np.asarray(full_generator_per_s, dtype=float)
    if (
        charts.shape != (n_cells, _N_FIELDS)
        or np.any(~np.isfinite(charts))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
        or full.shape != (n_reduced, n_reduced)
        or np.any(~np.isfinite(full))
    ):
        raise ValueError("generator block-decomposition inputs are invalid")

    state = causal_five_field_state_from_primitives(context, charts)
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    row_scales = np.asarray(
        scaling.row_scales[:n_reduced],
        dtype=float,
    )
    base_components = causal_five_field_stationary_residual_components(
        context,
        charts,
    )
    base_sum = np.sum(
        np.asarray(list(base_components.values()), dtype=float),
        axis=0,
    )
    base_residual_defect = _relative_maximum_defect(
        base_sum,
        evaluation.conservation_rows,
    )

    component_jacobians, color_count = (
        _colored_stationary_component_jacobians(
            context,
            charts,
            scales,
            row_scales,
            finite_difference_step=finite_difference_step,
            stencil_radius=stencil_radius,
        )
    )
    stationary_sum = np.sum(
        np.asarray(
            [matrix.toarray() for matrix in component_jacobians.values()],
            dtype=float,
        ),
        axis=0,
    )
    independent = causal_five_field_reduced_stationary_jacobian(
        context,
        vector,
        finite_difference_step=finite_difference_step,
    )
    independent_scales = np.asarray(
        independent["primitive_column_scales"],
        dtype=float,
    )
    independent_rows = np.asarray(
        independent["conservation_row_scales"],
        dtype=float,
    )
    if not np.allclose(
        independent_rows,
        row_scales,
        rtol=2.0e-14,
        atol=0.0,
    ):
        raise RuntimeError(
            "independent stationary row scaling changed"
        )
    stationary_reference = np.asarray(
        independent["stationary_reduced_scaled_jacobian"],
        dtype=float,
    ) * (scales / independent_scales)[None, :]
    stationary_jacobian_defect = _relative_maximum_defect(
        stationary_sum,
        stationary_reference,
    )

    storage = causal_five_field_reduced_storage_matrices(
        context,
        charts.ravel(),
        primitive_column_scales=scales,
        conservation_row_scales=row_scales,
        finite_difference_step=finite_difference_step,
        storage_quadrature_order=storage_quadrature_order,
        storage_directional_step=storage_directional_step,
    )
    mass = np.asarray(
        storage["descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    stationary_residual = (
        causal_five_field_reduced_stationary_residual(
            charts.ravel(),
            context,
        )
        / row_scales
    )
    solved_scaled_rate, rate_solve_defect = _sparse_mass_solve(
        mass,
        -stationary_residual,
    )
    if primitive_rate_per_s is None:
        scaled_rate = solved_scaled_rate
        physical_rate = scales * scaled_rate
    else:
        physical_rate = np.asarray(
            primitive_rate_per_s,
            dtype=float,
        )
        if physical_rate.shape == (n_cells, _N_FIELDS):
            physical_rate = physical_rate.ravel()
        if (
            physical_rate.shape != (n_reduced,)
            or np.any(~np.isfinite(physical_rate))
        ):
            raise ValueError(
                "supplied primitive rate has the wrong shape or value"
            )
        scaled_rate = physical_rate / scales
    storage_rate = causal_five_field_reduced_storage_rate_derivatives(
        context,
        charts.ravel(),
        physical_rate,
        primitive_column_scales=scales,
        conservation_row_scales=row_scales,
        storage_matrix_difference_step=finite_difference_step,
        storage_rate_derivative_step=storage_rate_derivative_step,
        storage_difference_step=storage_difference_step,
        storage_quadrature_order=storage_quadrature_order,
        storage_directional_step=storage_directional_step,
        backend="nested_matrix",
    )

    blocks: dict[str, np.ndarray] = {}
    maximum_solve_defect = rate_solve_defect
    for name, matrix in component_jacobians.items():
        solution, defect = _sparse_mass_solve(
            mass,
            matrix.toarray(),
        )
        blocks[name] = -solution
        maximum_solve_defect = max(maximum_solve_defect, defect)
    for name, key in (
        (
            "descriptor_mapped_rate_dependence",
            "conserved_storage_rate_derivative_scaled_matrix",
        ),
        (
            "descriptor_vertical_rate_dependence",
            "vertical_storage_rate_derivative_scaled_matrix",
        ),
    ):
        solution, defect = _sparse_mass_solve(
            mass,
            np.asarray(storage_rate[key], dtype=float),
        )
        blocks[name] = -solution
        maximum_solve_defect = max(maximum_solve_defect, defect)

    reconstructed = np.sum(
        np.asarray(list(blocks.values()), dtype=float),
        axis=0,
    )
    pre_remainder_defect = _relative_maximum_defect(
        reconstructed,
        full,
    )
    remainder = full - reconstructed
    blocks["residual_unattributed"] = remainder
    final = reconstructed + remainder
    final_defect = _relative_maximum_defect(final, full)
    remainder_norm = float(
        np.linalg.norm(remainder)
        / max(np.linalg.norm(full), np.finfo(float).tiny)
    )
    return CausalGeneratorBlockDecomposition(
        full_generator_per_s=np.array(full, copy=True),
        generator_blocks_per_s=blocks,
        descriptor_matrix=mass,
        scaled_primitive_rate_per_s=scaled_rate.reshape(
            n_cells,
            _N_FIELDS,
        ),
        physical_primitive_rate_per_s=physical_rate.reshape(
            n_cells,
            _N_FIELDS,
        ),
        component_names=tuple(blocks),
        reduced_pattern_colors=color_count,
        maximum_base_residual_reconstruction_defect=(
            base_residual_defect
        ),
        maximum_stationary_jacobian_reconstruction_defect=(
            stationary_jacobian_defect
        ),
        maximum_generator_reconstruction_defect_before_remainder=(
            pre_remainder_defect
        ),
        maximum_generator_reconstruction_defect_after_remainder=(
            final_defect
        ),
        maximum_mass_solve_relative_defect=maximum_solve_defect,
        residual_unattributed_relative_frobenius_norm=remainder_norm,
    )


def _block_diagonal_matrix(blocks: np.ndarray) -> np.ndarray:
    values = np.asarray(blocks, dtype=float)
    if (
        values.ndim != 3
        or values.shape[1:] != (_N_FIELDS, _N_FIELDS)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("block-diagonal matrix inputs are invalid")
    result = np.zeros(
        (_N_FIELDS * values.shape[0],) * 2,
        dtype=float,
    )
    for cell, block in enumerate(values):
        local = slice(
            _N_FIELDS * cell,
            _N_FIELDS * (cell + 1),
        )
        result[local, local] = block
    return result


def causal_five_field_scaled_shear_energy_operators(
    projectors: CausalShearEnergyProjectors,
    primitive_column_scales: np.ndarray,
    cell_measures: np.ndarray,
    *,
    family: str,
    cell_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return global scaled energy metrics and projector partitions."""

    family_index = _family_index(family)
    scales = np.asarray(primitive_column_scales, dtype=float)
    measures = np.asarray(cell_measures, dtype=float)
    n_cells = int(measures.size)
    if scales.shape == (n_cells * _N_FIELDS,):
        scales = scales.reshape(n_cells, _N_FIELDS)
    if (
        scales.shape != (n_cells, _N_FIELDS)
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
        or np.any(~np.isfinite(measures))
        or np.any(measures <= 0.0)
        or projectors.primitive_energy_grams.shape
        != (n_cells, _N_FIELDS, _N_FIELDS)
    ):
        raise ValueError("scaled shear-energy operator inputs are invalid")
    if cell_mask is None:
        mask = np.ones(n_cells, dtype=float)
    else:
        mask = np.asarray(cell_mask, dtype=float)
        if (
            mask.shape != (n_cells,)
            or np.any(~np.isfinite(mask))
            or np.any(mask < 0.0)
        ):
            raise ValueError("shear-energy cell mask is invalid")

    def scale_grams(grams: np.ndarray) -> np.ndarray:
        return (
            (measures * mask)[:, None, None]
            * scales[:, :, None]
            * np.asarray(grams, dtype=float)
            * scales[:, None, :]
        )

    inverse_scales = 1.0 / scales

    def scale_projectors(values: np.ndarray) -> np.ndarray:
        return (
            inverse_scales[:, :, None]
            * np.asarray(values, dtype=float)
            * scales[:, None, :]
        )

    shear_projector = _block_diagonal_matrix(
        scale_projectors(projectors.primitive_shear_projectors)
    )
    selected_projector = _block_diagonal_matrix(
        scale_projectors(
            projectors.primitive_family_projectors[family_index]
        )
    )
    complement_projector = _block_diagonal_matrix(
        scale_projectors(
            projectors.primitive_family_complement_projectors[
                family_index
            ]
        )
    )
    dimension = n_cells * _N_FIELDS
    non_shear_projector = np.eye(dimension) - shear_projector
    return {
        "total_energy_gram": _block_diagonal_matrix(
            scale_grams(projectors.primitive_energy_grams)
        ),
        "selected_energy_gram": _block_diagonal_matrix(
            scale_grams(
                projectors.primitive_family_energy_grams[
                    family_index
                ]
            )
        ),
        "complement_energy_gram": _block_diagonal_matrix(
            scale_grams(
                projectors.primitive_family_complement_energy_grams[
                    family_index
                ]
            )
        ),
        "shear_projector": shear_projector,
        "selected_projector": selected_projector,
        "complement_projector": complement_projector,
        "non_shear_projector": non_shear_projector,
    }


def _quadratic_history(
    state: np.ndarray,
    matrix: np.ndarray,
    *,
    factor: float,
) -> np.ndarray:
    return factor * np.einsum(
        "ti,ij,tj->t",
        state,
        matrix,
        state,
        optimize=True,
    )


def _cumulative_trapezoid(
    times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5
        * (values[:-1] + values[1:])
        * np.diff(times)
    )
    return result


def _integrated_ledger_defect(
    energy: np.ndarray,
    integral: np.ndarray,
) -> float:
    change = np.asarray(energy, dtype=float) - float(energy[0])
    scale = max(
        float(np.max(np.abs(change))),
        float(np.max(np.abs(integral))),
        abs(float(energy[0])),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(change - integral)) / scale)


def causal_five_field_shear_energy_ledger(
    full_generator_per_s: np.ndarray,
    generator_blocks_per_s: dict[str, np.ndarray],
    state_history: np.ndarray,
    times_seconds: np.ndarray,
    energy_operators: dict[str, np.ndarray],
    *,
    family: str,
) -> CausalShearEnergyLedger:
    """Evaluate an exact instantaneous and integrated shear-energy ledger."""

    _family_index(family)
    full = np.asarray(full_generator_per_s, dtype=float)
    state = np.asarray(state_history, dtype=float)
    if state.ndim == 3 and state.shape[2] == _N_FIELDS:
        state = state.reshape(state.shape[0], -1)
    times = np.asarray(times_seconds, dtype=float)
    if (
        full.ndim != 2
        or full.shape[0] != full.shape[1]
        or state.ndim != 2
        or state.shape[1] != full.shape[0]
        or times.shape != (state.shape[0],)
        or np.any(~np.isfinite(full))
        or np.any(~np.isfinite(state))
        or np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or not generator_blocks_per_s
    ):
        raise ValueError("shear-energy ledger inputs are invalid")
    blocks = {
        str(name): np.asarray(matrix, dtype=float)
        for name, matrix in generator_blocks_per_s.items()
    }
    if any(
        matrix.shape != full.shape or np.any(~np.isfinite(matrix))
        for matrix in blocks.values()
    ):
        raise ValueError("shear-energy generator block is invalid")
    reconstructed = np.sum(
        np.asarray(list(blocks.values()), dtype=float),
        axis=0,
    )
    if _relative_maximum_defect(reconstructed, full) > 1.0e-10:
        raise ValueError("shear-energy generator blocks do not close")

    total_gram = np.asarray(
        energy_operators["total_energy_gram"],
        dtype=float,
    )
    selected_gram = np.asarray(
        energy_operators["selected_energy_gram"],
        dtype=float,
    )
    complement_gram = np.asarray(
        energy_operators["complement_energy_gram"],
        dtype=float,
    )
    selected_projector = np.asarray(
        energy_operators["selected_projector"],
        dtype=float,
    )
    complement_projector = np.asarray(
        energy_operators["complement_projector"],
        dtype=float,
    )
    shear_projector = np.asarray(
        energy_operators["shear_projector"],
        dtype=float,
    )
    non_shear_projector = np.asarray(
        energy_operators["non_shear_projector"],
        dtype=float,
    )
    for matrix in (
        total_gram,
        selected_gram,
        complement_gram,
        selected_projector,
        complement_projector,
        shear_projector,
        non_shear_projector,
    ):
        if matrix.shape != full.shape or np.any(~np.isfinite(matrix)):
            raise ValueError("shear-energy operator shape is invalid")

    total_energy = _quadratic_history(
        state,
        total_gram,
        factor=0.5,
    )
    selected_energy = _quadratic_history(
        state,
        selected_gram,
        factor=0.5,
    )
    complement_energy = _quadratic_history(
        state,
        complement_gram,
        factor=0.5,
    )

    def rate(matrix: np.ndarray, generator: np.ndarray) -> np.ndarray:
        return np.einsum(
            "ti,ij,jk,tk->t",
            state,
            matrix,
            generator,
            state,
            optimize=True,
        )

    total_rate = rate(total_gram, full)
    selected_rate = rate(selected_gram, full)
    complement_rate = rate(complement_gram, full)
    total_by_block = {
        name: rate(total_gram, generator)
        for name, generator in blocks.items()
    }
    selected_by_block = {
        name: rate(selected_gram, generator)
        for name, generator in blocks.items()
    }
    complement_by_block = {
        name: rate(complement_gram, generator)
        for name, generator in blocks.items()
    }

    partitions = {
        "selected": selected_projector,
        "orthogonal_shear_complement": complement_projector,
        "non_shear": non_shear_projector,
    }

    def rate_by_source(
        metric: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            name: np.einsum(
                "ti,ij,jk,kl,tl->t",
                state,
                metric,
                full,
                projector,
                state,
                optimize=True,
            )
            for name, projector in partitions.items()
        }

    selected_by_source = rate_by_source(selected_gram)
    total_by_source = rate_by_source(total_gram)
    preserving = (
        selected_projector @ full @ selected_projector
        + complement_projector @ full @ complement_projector
        + non_shear_projector @ full @ non_shear_projector
    )
    transfer = full - preserving
    preserving_total = rate(total_gram, preserving)
    transfer_total = rate(total_gram, transfer)
    preserving_selected = rate(selected_gram, preserving)
    transfer_selected = rate(selected_gram, transfer)

    cumulative_total = _cumulative_trapezoid(times, total_rate)
    cumulative_selected = _cumulative_trapezoid(times, selected_rate)
    cumulative_complement = _cumulative_trapezoid(
        times,
        complement_rate,
    )
    energy_partition_scale = max(
        float(np.max(np.abs(total_energy))),
        np.finfo(float).tiny,
    )
    block_scale = max(
        float(np.max(np.abs(total_rate))),
        np.finfo(float).tiny,
    )
    source_scale = max(
        float(np.max(np.abs(selected_rate))),
        np.finfo(float).tiny,
    )
    return CausalShearEnergyLedger(
        family=str(family),
        times_seconds=np.array(times, copy=True),
        total_energy=total_energy,
        selected_energy=selected_energy,
        complement_energy=complement_energy,
        total_energy_rate_per_s=total_rate,
        selected_energy_rate_per_s=selected_rate,
        complement_energy_rate_per_s=complement_rate,
        total_rate_by_block_per_s=total_by_block,
        selected_rate_by_block_per_s=selected_by_block,
        complement_rate_by_block_per_s=complement_by_block,
        selected_rate_by_source_partition_per_s=selected_by_source,
        total_rate_by_source_partition_per_s=total_by_source,
        preserving_total_rate_per_s=preserving_total,
        transfer_total_rate_per_s=transfer_total,
        preserving_selected_rate_per_s=preserving_selected,
        transfer_selected_rate_per_s=transfer_selected,
        cumulative_total_rate_integral=cumulative_total,
        cumulative_selected_rate_integral=cumulative_selected,
        cumulative_complement_rate_integral=cumulative_complement,
        maximum_instantaneous_energy_partition_defect=float(
            np.max(
                np.abs(
                    total_energy
                    - selected_energy
                    - complement_energy
                )
            )
            / energy_partition_scale
        ),
        maximum_instantaneous_block_ledger_defect=float(
            np.max(
                np.abs(
                    total_rate
                    - np.sum(
                        np.asarray(
                            list(total_by_block.values()),
                            dtype=float,
                        ),
                        axis=0,
                    )
                )
            )
            / block_scale
        ),
        maximum_instantaneous_source_partition_defect=float(
            np.max(
                np.abs(
                    selected_rate
                    - np.sum(
                        np.asarray(
                            list(selected_by_source.values()),
                            dtype=float,
                        ),
                        axis=0,
                    )
                )
            )
            / source_scale
        ),
        maximum_integrated_total_ledger_defect=(
            _integrated_ledger_defect(total_energy, cumulative_total)
        ),
        maximum_integrated_selected_ledger_defect=(
            _integrated_ledger_defect(
                selected_energy,
                cumulative_selected,
            )
        ),
        maximum_integrated_complement_ledger_defect=(
            _integrated_ledger_defect(
                complement_energy,
                cumulative_complement,
            )
        ),
    )
