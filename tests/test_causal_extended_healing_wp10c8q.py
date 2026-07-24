from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_causal_extended_healing_wp10c8q import (  # noqa: E402
    INTERFACE_FLUX_RELATIVE_GATE,
    _continuation_reference_window,
    _flat_interface_mje_half_difference,
    _held_out_slow_rate_direction,
    _merge_fixed_result_rows,
    _normalized_mje_interface_half_difference,
)

from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_central_perfect_flux_from_face_charts,
    causal_five_field_face_flux_decomposition,
    causal_five_field_reconstruct_face_charts,
    causal_internal_face_boundary_rates,
    causal_path_integrated_component_decomposition,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)


def test_internal_face_boundary_rates_are_conservative() -> None:
    faces = np.asarray(
        [
            [1.0, 2.0],
            [-3.0, 5.0],
        ]
    )

    shells = causal_internal_face_boundary_rates(faces)

    np.testing.assert_array_equal(
        shells,
        np.asarray(
            [
                [1.0, 2.0],
                [-4.0, 3.0],
                [3.0, -5.0],
            ]
        ),
    )
    np.testing.assert_array_equal(np.sum(shells, axis=0), 0.0)


def test_path_component_decomposition_closes_nonlinear_change() -> None:
    def output(values: np.ndarray) -> np.ndarray:
        x, y, z = values
        return np.asarray(
            [x * x + x * y + 0.5 * z, y * y - x * z]
        )

    initial = np.asarray([0.7, -0.2, 1.1])
    final = np.asarray([1.3, 0.5, -0.4])

    audit = causal_path_integrated_component_decomposition(
        output,
        initial,
        final,
        quadrature_order=8,
        finite_difference_relative_step=1.0e-6,
    )

    np.testing.assert_allclose(
        audit.reconstructed_difference,
        audit.endpoint_difference,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    assert audit.maximum_reconstruction_relative_defect <= 2.0e-10
    assert audit.maximum_quadrature_relative_defect <= 2.0e-10


def test_central_perfect_face_helper_reconstructs_production_split() -> None:
    context = make_causal_five_field_regression_context(
        6,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_rate_scheme="arithmetic_face",
        cell_source_quadrature="gauss_legendre_4_local_rates",
        cell_storage_quadrature="gauss_legendre_4",
    )
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        state.primitives,
    )
    split = causal_five_field_face_flux_decomposition(context, vector)

    rebuilt = np.asarray(
        [
            causal_five_field_central_perfect_flux_from_face_charts(
                context,
                face,
                reconstruction.left_face_charts[face],
                reconstruction.right_face_charts[face],
            )
            for face in range(1, state.n_cells)
        ]
    )

    np.testing.assert_allclose(
        rebuilt,
        split.central_perfect_weighted_face_fluxes_over_c,
        rtol=2.0e-14,
        atol=1.0e-14,
    )


def test_segmented_fixed_result_rows_preserve_work_and_ledger() -> None:
    def row(multiplier: float) -> dict:
        return {
            "passed": True,
            "message": "ok",
            "subdivisions": 2,
            "timestep_seconds": 0.01,
            "completed_steps": 2,
            "bdf1_steps": 0,
            "bdf2_steps": 2,
            "maximum_scaled_residual": multiplier,
            "maximum_scaled_algebraic_residual": 0.5 * multiplier,
            "maximum_scaled_primitive_change": 0.25 * multiplier,
            "maximum_scaled_total_change": 0.125 * multiplier,
            "maximum_discrete_ledger_relative_defect": 1.0e-12,
            "cumulative_physical_ledger_relative_defect": 1.0e-12,
            "maximum_linear_residual": 1.0e-13,
            "maximum_newton_iterations": int(multiplier),
            "function_evaluations": 10,
            "jacobian_evaluations": 2,
            "newton_iterations": 4,
            "state_gates": {"passed": True, "marker": multiplier},
            "cumulative_physical_ledger": {
                "actual_conserved_storage": np.asarray(
                    [multiplier, 0.0, 0.0]
                ),
                "actual_vertical_storage": np.zeros(3),
                "trapezoidal_boundary_transport": np.asarray(
                    [-multiplier, 0.0, 0.0]
                ),
                "trapezoidal_endogenous_source": np.zeros(3),
                "exact_prescribed_stream_source": np.zeros(3),
                "closure_defect": np.zeros(3),
            },
            "wall_seconds": 3.0,
        }

    merged = _merge_fixed_result_rows([row(1.0), row(2.0)])

    assert merged["subdivisions"] == 4
    assert merged["completed_steps"] == 4
    assert merged["function_evaluations"] == 20
    assert merged["jacobian_evaluations"] == 4
    assert merged["maximum_scaled_residual"] == 2.0
    assert merged["state_gates"]["marker"] == 2.0
    assert merged["cumulative_physical_ledger_relative_defect"] == 0.0
    np.testing.assert_array_equal(
        merged["cumulative_physical_ledger"]["actual_conserved_storage"],
        np.asarray([3.0, 0.0, 0.0]),
    )


def test_all_interface_normalization_selects_mje_without_broadcasting_faces() -> None:
    minus = np.zeros((2, 4, 5))
    plus = np.zeros_like(minus)
    plus[..., 0] = 2.0
    plus[..., 1] = 1.0e9
    plus[..., 2] = 4.0
    plus[..., 3] = 6.0
    plus[..., 4] = -1.0e9
    scales = np.arange(1.0, 13.0)

    normalized = _normalized_mje_interface_half_difference(
        plus,
        minus,
        scales,
    )

    assert normalized.shape == (2, 4, 3)
    expected = (
        np.asarray([1.0, 2.0, 3.0])[None, None, :]
        / (
            INTERFACE_FLUX_RELATIVE_GATE
            * scales.reshape(4, 3)[None, :, :]
        )
    )
    np.testing.assert_array_equal(
        normalized,
        np.broadcast_to(expected, normalized.shape),
    )


def test_replay_reference_window_uses_absolute_saved_time_index() -> None:
    reference = np.arange(101 * 2, dtype=float).reshape(101, 2)

    window = _continuation_reference_window(
        reference,
        restart_elapsed_time=0.10,
        timestep=0.00125,
        replay_steps=20,
    )

    np.testing.assert_array_equal(window, reference[80:101])


def test_flat_interface_output_is_reshaped_before_face_selection() -> None:
    minus = np.zeros(12)
    plus = 2.0 * np.arange(1.0, 13.0)

    half = _flat_interface_mje_half_difference(plus, minus)

    assert half.shape == (4, 3)
    np.testing.assert_array_equal(half[3], np.asarray([10.0, 11.0, 12.0]))


def test_held_out_rate_direction_is_weighted_orthogonal() -> None:
    audit = SimpleNamespace(
        null_basis_audit=SimpleNamespace(
            basis=np.eye(3),
            state_weights=np.ones(3),
        ),
        gate_normalized_null_operator=np.diag([3.0, 2.0, 1.0]),
    )

    held_out, diagnostics = _held_out_slow_rate_direction(
        audit,
        np.asarray([1.0, 0.0, 0.0]),
        np.ones(3),
    )

    np.testing.assert_allclose(held_out, np.asarray([0.0, 1.0, 0.0]))
    assert diagnostics["weighted_orthogonality_defect"] <= 1.0e-15
