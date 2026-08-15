from __future__ import annotations

from dataclasses import replace

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CausalFiveFieldMonolithicBDFRestart,
    advance_causal_five_field_monolithic_bdf,
    causal_five_field_monolithic_bdf_history,
    causal_five_field_monolithic_bdf_restarts_equal,
    causal_five_field_monolithic_frozen_tangent,
    causal_five_field_monolithic_storage_increment,
    causal_five_field_dae_scaling,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_monolithic_bdf,
    evaluate_causal_five_field_dae,
    load_causal_five_field_monolithic_bdf_restart,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
    save_causal_five_field_monolithic_bdf_restart,
)


def _problem(n_cells: int = 5, *, with_tangent: bool = False):
    context = make_causal_five_field_regression_context(n_cells)
    base = np.asarray(
        make_causal_five_field_seed(context).primitives,
        dtype=float,
    )
    context = replace(
        context,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_storage_quadrature=(
            "midpoint" if with_tangent else "gauss_legendre_4"
        ),
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.array(base[-1], copy=True),
    ).validated()
    state = causal_five_field_state_from_primitives(context, base)
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    scaling = causal_five_field_dae_scaling(state, evaluation)
    dimensions = base.size
    columns = np.asarray(
        scaling.column_scales[dimensions : 2 * dimensions],
        dtype=float,
    )
    rows = np.asarray(
        scaling.row_scales[:dimensions],
        dtype=float,
    )
    tangent = (
        causal_five_field_monolithic_frozen_tangent(
            context,
            base,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
        )
        if with_tangent
        else None
    )
    return context, base, tangent


def _increment(base: np.ndarray, scale: float = 1.0e-7) -> np.ndarray:
    envelope = np.sin(np.linspace(0.0, np.pi, base.shape[0]))[:, None]
    vector = np.asarray([1.0, -0.5, 0.25, -0.125, 0.5])[None, :]
    field_scales = np.maximum(np.max(np.abs(base), axis=0), 1.0e-12)
    return scale * envelope * vector * field_scales[None, :]


def test_monolithic_bdf2_uses_stored_complete_path_increment() -> None:
    context, base, _tangent = _problem()
    first_increment = _increment(base)
    first_storage = causal_five_field_monolithic_storage_increment(
        context,
        base,
        base + first_increment,
    )
    history = causal_five_field_monolithic_bdf_history(
        first_increment,
        first_storage,
        1.0e-5,
    )
    second_increment = 0.75 * first_increment
    evaluation = evaluate_causal_five_field_monolithic_bdf(
        base + first_increment,
        base + first_increment + second_increment,
        1.0e-5,
        context,
        order=2,
        history=history,
    )
    np.testing.assert_allclose(
        evaluation.weighted_complete_storage_increment,
        (
            1.5
            * (
                evaluation.current_storage_increment.mapped_path_increment
                + evaluation.current_storage_increment
                .responsive_height_path_increment
            )
            - 0.5 * history.previous_complete_storage_increment
        ),
        rtol=5.0e-16,
        atol=0.0,
    )
    assert evaluation.maximum_block_ledger_defect == 0.0
    assert evaluation.mapped_storage_uses_stable_exact_path_integral
    assert evaluation.incoming_excision_characteristics == 0


def test_monolithic_bdf2_direct_storage_rate_matches_increment_form() -> None:
    context, base, _tangent = _problem()
    timestep = 1.0e-5
    first_increment = _increment(base)
    first_storage = causal_five_field_monolithic_storage_increment(
        context,
        base,
        base + first_increment,
    )
    history = causal_five_field_monolithic_bdf_history(
        first_increment,
        first_storage,
        timestep,
    )
    old = base + first_increment
    new = old + 0.75 * first_increment
    interval_rate = (new - old) / timestep
    increment_form = evaluate_causal_five_field_monolithic_bdf(
        old,
        new,
        timestep,
        context,
        order=2,
        history=history,
    )
    direct_rate = evaluate_causal_five_field_monolithic_bdf(
        old,
        new,
        timestep,
        context,
        order=2,
        history=history,
        current_primitive_rate_per_s=interval_rate,
    )
    scale = max(
        float(np.linalg.norm(increment_form.residual_rows)),
        float(np.linalg.norm(direct_rate.residual_rows)),
        np.finfo(float).tiny,
    )
    assert (
        np.linalg.norm(
            increment_form.residual_rows - direct_rate.residual_rows
        )
        / scale
        <= 1.0e-9
    )
    assert direct_rate.temporal_storage_uses_direct_rate_action


def test_monolithic_bdf1_and_bdf2_reach_frozen_method_gate() -> None:
    context, base, tangent = _problem(5, with_tangent=True)
    assert tangent is not None
    first = advance_causal_five_field_monolithic_bdf(
        context,
        base,
        1.0e-5,
        tangent,
        order=1,
        residual_tolerance=1.0e-8,
        maximum_scaled_primitive_change=2.0e-2,
    )
    assert first.accepted, first.message
    assert first.maximum_scaled_residual <= 1.0e-8
    assert first.maximum_discrete_ledger_defect <= 1.0e-12
    assert first.history is not None

    second = advance_causal_five_field_monolithic_bdf(
        context,
        first.primitive_charts,
        1.0e-5,
        tangent,
        order=2,
        history=first.history,
        residual_tolerance=1.0e-8,
        maximum_scaled_primitive_change=2.0e-2,
    )
    assert second.accepted, second.message
    assert second.maximum_scaled_residual <= 1.0e-8
    assert second.maximum_scaled_algebraic_residual == 0.0
    assert second.history is not None


def test_monolithic_bdf_restart_round_trips_bitwise(tmp_path) -> None:
    context, base, _tangent = _problem()
    increment = _increment(base)
    storage = causal_five_field_monolithic_storage_increment(
        context,
        base,
        base + increment,
    )
    restart = CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=base + increment,
        history=causal_five_field_monolithic_bdf_history(
            increment,
            storage,
            1.0e-5,
        ),
        elapsed_time_seconds=1.0e-5,
        completed_steps=1,
        next_order=2,
        provenance={"work_package": "unit_test"},
    )
    path = tmp_path / "restart.npz"
    save_causal_five_field_monolithic_bdf_restart(
        path,
        context,
        restart,
    )
    restored = load_causal_five_field_monolithic_bdf_restart(
        path,
        context,
        expected_provenance=restart.provenance,
    )
    assert causal_five_field_monolithic_bdf_restarts_equal(
        restart,
        restored,
    )
