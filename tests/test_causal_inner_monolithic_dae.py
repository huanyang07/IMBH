from __future__ import annotations

from dataclasses import replace

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_monolithic_storage_increment,
    causal_five_field_temporal_storage_integrability_audit,
    evaluate_causal_five_field_monolithic_backward_euler,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
)


def _context_and_charts(n_cells: int = 5):
    context = make_causal_five_field_regression_context(n_cells)
    charts = np.asarray(
        make_causal_five_field_seed(context).primitives,
        dtype=float,
    )
    context = replace(
        context,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_storage_quadrature="gauss_legendre_4",
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.array(
            charts[-1],
            copy=True,
        ),
    ).validated()
    return context, charts


def _small_increment(charts: np.ndarray) -> np.ndarray:
    cells = charts.shape[0]
    phase = np.sin(np.linspace(0.0, np.pi, cells))[:, None]
    scale = np.asarray(
        [1.0e-5, 0.0, 0.0, 1.0e-5, 0.0],
        dtype=float,
    )
    return phase * scale[None, :]


def test_monolithic_storage_is_zero_and_reversible() -> None:
    context, charts = _context_and_charts()
    zero = causal_five_field_monolithic_storage_increment(
        context,
        charts,
        charts,
    )
    assert np.array_equal(
        zero.total_storage_increment,
        np.zeros_like(charts),
    )
    assert zero.maximum_mapped_path_closure_defect == 0.0

    new = charts + _small_increment(charts)
    forward = causal_five_field_monolithic_storage_increment(
        context,
        charts,
        new,
        temporal_quadrature_order=6,
    )
    backward = causal_five_field_monolithic_storage_increment(
        context,
        new,
        charts,
        temporal_quadrature_order=6,
    )
    scale = max(
        float(np.linalg.norm(forward.total_storage_increment)),
        np.finfo(float).tiny,
    )
    reversal = np.linalg.norm(
        forward.total_storage_increment
        + backward.total_storage_increment
    ) / scale
    assert forward.maximum_mapped_path_closure_defect <= 2.0e-8
    assert forward.maximum_affine_reconstruction_path_defect <= 1.0e-12
    assert reversal <= 2.0e-10
    assert forward.one_flux_reconstruction_for_space_and_storage
    assert forward.uses_exact_affine_reconstruction_path_derivative
    assert forward.mapped_storage_is_exact_endpoint_increment
    assert forward.responsive_height_is_nonconservative_temporal_product
    assert forward.minimum_path_reconstruction_factor == 1.0
    assert forward.maximum_path_reconstruction_factor_change == 0.0


def test_responsive_height_storage_is_a_nonexact_temporal_one_form() -> None:
    context, charts = _context_and_charts()
    audit = causal_five_field_temporal_storage_integrability_audit(
        context,
        float(context.grid.centers[0]),
        charts[0],
    )
    assert audit.loop_fields == (2, 3)
    assert audit.relative_vertical_exterior_derivative > 0.1
    assert audit.relative_complete_exterior_derivative > 1.0e-7
    assert audit.relative_loop_to_vertical_path > 1.0e-5
    assert np.linalg.norm(audit.loop_vertical_increment) > 0.0


def test_monolithic_residual_has_one_storage_and_spatial_ledger() -> None:
    context, charts = _context_and_charts()
    new = charts + _small_increment(charts)
    evaluation = evaluate_causal_five_field_monolithic_backward_euler(
        charts,
        new,
        1.0e-4,
        context,
        temporal_quadrature_order=6,
    )
    blocks = (
        evaluation.mapped_temporal_storage_rows,
        evaluation.responsive_height_temporal_storage_rows,
        evaluation.conservative_transport_rows,
        evaluation.shear_principal_rows,
        evaluation.height_principal_rows,
        evaluation.local_stress_relaxation_rows,
        evaluation.geometry_rows,
        evaluation.cooling_rows,
        evaluation.stream_rows,
        evaluation.lower_height_work_rows,
    )
    np.testing.assert_allclose(
        evaluation.residual_rows,
        np.sum(np.asarray(blocks), axis=0),
        rtol=0.0,
        atol=0.0,
    )
    faces = (
        evaluation.stationary_ledger.interfaces
        .candidate_shared_face_fluxes_over_c
    )
    np.testing.assert_allclose(
        evaluation.conservative_transport_rows,
        faces[1:] - faces[:-1],
        rtol=0.0,
        atol=0.0,
    )
    assert evaluation.maximum_block_ledger_defect == 0.0
    assert evaluation.incoming_excision_characteristics == 0
    assert not evaluation.uses_production_generator
    assert not evaluation.uses_production_anchor_storage_derivative
    assert evaluation.maximum_center_broken_path_adjustment <= 2.0e-8
