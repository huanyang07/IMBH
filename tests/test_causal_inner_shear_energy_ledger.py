from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply

from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_evolving_tangent_matrices,
    causal_five_field_generator_block_decomposition,
    causal_five_field_scaled_shear_energy_operators,
    causal_five_field_shear_energy_ledger,
    causal_five_field_shear_energy_projectors,
    causal_five_field_state_from_primitives,
    causal_five_field_stationary_residual_components,
    evaluate_causal_five_field_dae,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)


def _regression_evolving():
    context = make_causal_five_field_regression_context(8)
    primitives = make_causal_five_field_seed(context).primitives
    state = causal_five_field_state_from_primitives(context, primitives)
    vector = pack_causal_five_field_state(state)
    evolving = causal_five_field_evolving_tangent_matrices(
        context,
        vector,
        primitive_rate_per_s=None,
        finite_difference_step=2.0e-6,
        descriptor_timestep_seconds=1.0,
        storage_difference_step=1.0e-4,
        storage_rate_derivative_step=2.0e-6,
        storage_quadrature_order=4,
        storage_directional_step=1.0e-3,
    )
    return context, primitives, evolving


def test_energy_orthogonal_family_projectors_partition_total_energy() -> None:
    context = make_causal_five_field_regression_context(8)
    primitives = make_causal_five_field_seed(context).primitives
    projectors = causal_five_field_shear_energy_projectors(
        context,
        primitives,
    )

    assert projectors.minimum_positive_energy_eigenvalue > 0.0
    assert projectors.maximum_shear_projector_defect <= 1.0e-8
    assert projectors.maximum_family_projector_defect <= 1.0e-8
    assert projectors.maximum_partition_defect <= 1.0e-8
    assert projectors.maximum_energy_self_adjoint_defect <= 1.0e-10
    assert projectors.maximum_energy_partition_defect <= 1.0e-10

    scales = np.maximum(np.abs(primitives), 1.0e-6)
    for family in ("inward_shear", "outward_shear"):
        operators = causal_five_field_scaled_shear_energy_operators(
            projectors,
            scales,
            context.grid.cell_measures,
            family=family,
        )
        np.testing.assert_allclose(
            operators["selected_energy_gram"]
            + operators["complement_energy_gram"],
            operators["total_energy_gram"],
            rtol=1.0e-10,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            operators["selected_projector"]
            + operators["complement_projector"]
            + operators["non_shear_projector"],
            np.eye(40),
            rtol=0.0,
            atol=1.0e-10,
        )


def test_stationary_physical_components_reconstruct_reduced_residual() -> None:
    context = make_causal_five_field_regression_context(8)
    primitives = make_causal_five_field_seed(context).primitives
    components = causal_five_field_stationary_residual_components(
        context,
        primitives,
    )
    reconstructed = np.sum(
        np.asarray(list(components.values())),
        axis=0,
    )
    state = causal_five_field_state_from_primitives(context, primitives)
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    np.testing.assert_allclose(
        reconstructed,
        evaluation.conservation_rows,
        rtol=2.0e-13,
        atol=1.0e-13,
    )
    assert "transport_rusanov" in components
    assert "transport_inner_boundary" in components
    assert any(name.startswith("source_") for name in components)


def test_generator_blocks_and_quadratic_energy_ledgers_close() -> None:
    context, primitives, evolving = _regression_evolving()
    scales = np.asarray(
        evolving["primitive_column_scales"],
        dtype=float,
    )
    full = np.asarray(
        evolving["evolving_scaled_generator_per_s"],
        dtype=float,
    )
    decomposition = causal_five_field_generator_block_decomposition(
        context,
        primitives,
        primitive_column_scales=scales,
        full_generator_per_s=full,
        primitive_rate_per_s=np.asarray(
            evolving["primitive_rate_per_s"],
            dtype=float,
        ),
    )

    assert (
        decomposition.maximum_base_residual_reconstruction_defect
        <= 1.0e-12
    )
    assert (
        decomposition.maximum_stationary_jacobian_reconstruction_defect
        <= 2.0e-6
    )
    assert (
        decomposition.maximum_generator_reconstruction_defect_after_remainder
        <= 1.0e-14
    )
    assert decomposition.maximum_mass_solve_relative_defect <= 1.0e-10
    np.testing.assert_array_equal(
        decomposition.physical_primitive_rate_per_s,
        np.asarray(evolving["primitive_rate_per_s"], dtype=float),
    )
    np.testing.assert_allclose(
        np.sum(
            np.asarray(
                list(decomposition.generator_blocks_per_s.values())
            ),
            axis=0,
        ),
        full,
        rtol=0.0,
        atol=2.0e-12,
    )

    projectors = causal_five_field_shear_energy_projectors(
        context,
        primitives,
    )
    operators = causal_five_field_scaled_shear_energy_operators(
        projectors,
        scales,
        context.grid.cell_measures,
        family="inward_shear",
    )
    initial = np.linspace(-0.3, 0.4, full.shape[0])
    times = np.linspace(0.0, 2.0e-3, 41)
    history = np.asarray(
        expm_multiply(
            full,
            initial,
            start=times[0],
            stop=times[-1],
            num=times.size,
            endpoint=True,
        ),
        dtype=float,
    )
    ledger = causal_five_field_shear_energy_ledger(
        full,
        decomposition.generator_blocks_per_s,
        history,
        times,
        operators,
        family="inward_shear",
    )

    assert ledger.maximum_instantaneous_energy_partition_defect <= 1.0e-10
    assert ledger.maximum_instantaneous_block_ledger_defect <= 1.0e-10
    assert ledger.maximum_instantaneous_source_partition_defect <= 1.0e-10
    assert ledger.maximum_integrated_total_ledger_defect <= 1.0e-4
    assert ledger.maximum_integrated_selected_ledger_defect <= 1.0e-4
    assert ledger.maximum_integrated_complement_ledger_defect <= 1.0e-4
