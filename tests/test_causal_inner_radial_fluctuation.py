from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_radial_candidate_ledger,
    causal_five_field_radial_extended_path_jump,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import _cell_state


def test_radial_path_uses_actual_endpoint_measures_and_sign() -> None:
    context = make_causal_five_field_regression_context(8)
    primitives = make_causal_five_field_seed(context).primitives
    cell = 3
    chart = np.asarray(primitives[cell], dtype=float)
    lower = float(context.grid.edges[cell])
    upper = float(context.grid.edges[cell + 1])
    path = causal_five_field_radial_extended_path_jump(
        context,
        lower,
        upper,
        chart,
        chart,
    )
    lower_state = _cell_state(context, lower, chart)
    upper_state = _cell_state(context, upper, chart)
    expected = (
        upper_state.geometry.face_measure * upper_state.flux_over_c
        - lower_state.geometry.face_measure * lower_state.flux_over_c
    )
    np.testing.assert_allclose(
        path.conservative_endpoint_jump_over_c,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert np.array_equal(
        path.shear_source_path_integral_over_c,
        np.zeros(5),
    )
    assert np.array_equal(
        path.vertical_source_path_integral_over_c,
        np.zeros(5),
    )
    np.testing.assert_allclose(
        path.total_principal_jump_over_c,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert path.source_partition_defect == 0.0
    assert path.principal_closure_defect == 0.0


def test_radial_candidate_has_one_shared_flux_and_complete_block_ledger() -> None:
    context = make_causal_five_field_regression_context(8)
    primitives = make_causal_five_field_seed(context).primitives
    result = causal_five_field_radial_candidate_ledger(
        context,
        primitives,
    )
    blocks = (
        result.conservative_transport_rows,
        result.shear_principal_rows,
        result.height_principal_rows,
        result.local_stress_relaxation_rows,
        result.geometry_rows,
        result.cooling_rows,
        result.stream_rows,
        result.lower_height_work_rows,
    )
    np.testing.assert_allclose(
        result.residual_rows,
        np.sum(np.asarray(blocks), axis=0),
        rtol=0.0,
        atol=0.0,
    )
    candidate_fluxes = (
        result.interfaces.candidate_shared_face_fluxes_over_c
    )
    np.testing.assert_allclose(
        result.conservative_transport_rows,
        candidate_fluxes[1:] - candidate_fluxes[:-1],
        rtol=0.0,
        atol=0.0,
    )
    assert result.local_block_ledger_defect == 0.0
    assert result.source_double_count_defect == 0.0
    assert result.interfaces.incoming_excision_characteristics == 0
    assert all(
        np.array_equal(
            path.shear_source_path_integral_over_c[:4],
            np.zeros(4),
        )
        for path in result.within_cell_paths
    )
    assert all(
        path.vertical_source_path_integral_over_c[4] == 0.0
        for path in result.within_cell_paths
    )
    assert np.all(result.local_stress_relaxation_rows[:, :4] == 0.0)
    assert np.all(result.cooling_rows[:, 4] == 0.0)
    assert np.all(result.stream_rows[:, 4] == 0.0)
