from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i

from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (
    causal_five_field_rusanov_control_diagnostics,
)
from imri_qpe.layer3_minidisk_1d import (
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
)


def test_cross_mesh_scalar_agreement_uses_activity_floor() -> None:
    inactive = wp10c8i._scalar_cross_mesh_agreement(
        0.0,
        0.5 * wp10c8i.CROSS_MESH_GAIN_ACTIVITY_FLOOR,
    )
    active = wp10c8i._scalar_cross_mesh_agreement(
        0.0,
        2.0 * wp10c8i.CROSS_MESH_GAIN_ACTIVITY_FLOOR,
    )

    assert not inactive["comparison_active"]
    assert inactive["passed"]
    assert active["comparison_active"]
    assert not active["passed"]


def test_common_log_h_rows_include_every_fine_native_center(
    monkeypatch,
) -> None:
    def fake_response_stack(arrays, _metadata, _level_index):
        count = arrays["radius_rg"].size
        names = (
            "scientific",
            *(f"h_{index}" for index in range(count)),
            "interface",
            "rate",
        )
        blocks = {
            "scientific": (0, 1),
            "full_log_h_over_r": (1, 1 + count),
            "macro_interface_flux": (1 + count, 2 + count),
            "coarse_coordinate_rate": (2 + count, 3 + count),
        }
        return np.empty((3 + count, 1)), np.ones(3 + count), names, blocks

    monkeypatch.setattr(wp10c8i, "_response_stack", fake_response_stack)
    coarse_arrays = {
        "radius_rg": np.asarray([2.0, 4.0]),
        "grid_edges_rg": np.asarray([1.0, 5.0]),
    }
    fine_arrays = {
        "radius_rg": np.asarray([2.0, 3.0, 4.0]),
        "grid_edges_rg": np.asarray([1.0, 5.0]),
    }
    rows = wp10c8i._matched_output_rows(
        coarse_arrays,
        {},
        fine_arrays,
        {},
        0,
    )
    row_at_three = next(
        row
        for row in rows
        if row["block"] == "full_log_h_over_r"
        and row["common_radius_rg"] == 3.0
    )
    coarse_bounds = np.asarray([1.0, 0.1, 0.1, 1.0, 1.0])
    fine_bounds = np.asarray([1.0, 0.1, 2.0, 0.1, 1.0, 1.0])
    agreement = wp10c8i._matched_bound_agreement(
        rows,
        coarse_bounds,
        fine_bounds,
    )

    assert row_at_three["fine_indices"] == (2,)
    assert not row_at_three["binding_cross_mesh"]
    assert (
        agreement["diagnostic_controlling_output"]["common_radius_rg"]
        == 3.0
    )
    # Interpolation of optimized rowwise gains is diagnostic only.  The
    # native per-mesh maximum lower/upper comparison is performed separately
    # by the campaign.
    assert agreement["passed"]


def test_audit_summary_reports_distinct_upper_controller() -> None:
    audit = SimpleNamespace(
        per_output_maximum_gains=np.asarray([0.8, 0.7]),
        per_output_admissible_lower_gains=np.asarray([0.6, 0.5]),
        per_output_admissible_upper_gains=np.asarray([0.7, 0.9]),
        per_output_l2_maximum_pointwise_ratios=np.asarray([1.0, 1.0]),
        controlling_admissible_output_index=0,
        maximum_per_output_gain=0.8,
        maximum_admissible_lower_gain=0.6,
        maximum_admissible_upper_gain=0.9,
        maximum_gain=1.0,
        controlling_admissible_state_direction=np.asarray(
            [1.0, 0.0, 0.0, 0.0, 0.0]
        ),
        controlling_admissible_gate_normalized_output_response=np.asarray(
            [0.6, 0.0]
        ),
        null_basis_audit=SimpleNamespace(
            constraint_rank=1,
            active_row_count=1,
            nullity=1,
            condition_estimate=1.0,
            raw_constraint_defect=0.0,
            weighted_orthogonality_defect=0.0,
        ),
        admissible_leading_state_subspace=np.asarray(
            [[1.0], [0.0], [0.0], [0.0], [0.0]]
        ),
        admissible_leading_output_indices=np.asarray([0]),
        singular_values=np.asarray([1.0]),
    )

    summary, _traces = wp10c8i._audit_summary(
        audit,
        ("lower_controller", "upper_controller"),
        np.ones(5),
    )

    assert summary["controlling_admissible_lower_output"] == (
        "lower_controller"
    )
    assert summary["controlling_admissible_upper_output"] == (
        "upper_controller"
    )


def test_rusanov_control_diagnostics_cover_every_interior_face() -> None:
    context = make_causal_five_field_regression_context(4)
    state = make_causal_five_field_seed(context)

    audit = causal_five_field_rusanov_control_diagnostics(
        context,
        state.primitives,
    )

    assert audit["control_codes"].shape == (3,)
    assert len(audit["control_labels"]) == 3
    assert np.all(np.isfinite(audit["relative_control_margins"]))
    assert np.all(audit["relative_control_margins"] >= 0.0)
    assert audit["conserved_jumps"].shape == (3, 5)
    assert audit["relative_conserved_jump_l2"].shape == (3,)
    assert audit["relative_conserved_jump_maximum"].shape == (3,)
    assert audit["relative_scaled_conserved_jump_l2"].shape == (3,)
    assert audit[
        "relative_scaled_conserved_jump_maximum"
    ].shape == (3,)
    assert audit["exact_zero_conserved_jump"].shape == (3,)
    assert audit["second_control_codes"].shape == (3,)
    assert audit["candidate_absolute_speeds_over_c"].shape == (3, 10)


def _synthetic_rusanov_audit(
    *,
    margin: float,
    jump: float,
    code: int = 0,
) -> dict:
    return {
        "relative_control_margins": np.asarray([margin]),
        "relative_conserved_jump_l2": np.asarray([jump]),
        "relative_scaled_conserved_jump_maximum": np.asarray([jump]),
        "exact_zero_conserved_jump": np.asarray([jump == 0.0]),
        "control_codes": np.asarray([code], dtype=int),
    }


def test_rusanov_zero_jump_tie_is_resolved() -> None:
    result = wp10c8i._rusanov_branch_resolution(
        _synthetic_rusanov_audit(margin=0.0, jump=0.0)
    )

    assert result["passed"]
    assert result["exact_zero_jump_face_state_count"] == 1


def test_rusanov_finite_jump_tie_is_rejected() -> None:
    result = wp10c8i._rusanov_branch_resolution(
        _synthetic_rusanov_audit(margin=0.0, jump=2.0e-4)
    )

    assert not result["passed"]
    assert result["unresolved_face_state_count"] == 1


def test_numerically_unique_near_tie_does_not_bind_finite_time() -> None:
    result = wp10c8i._rusanov_branch_resolution(
        _synthetic_rusanov_audit(
            margin=1.0e-10,
            jump=2.0e-4,
        )
    )

    assert result["passed"]
    assert not result["declared_finite_branch_screen_passed"]
    assert (
        result[
            "declared_finite_branch_screen_unresolved_face_state_count"
        ]
        == 1
    )


def test_rusanov_control_switch_requires_jump_suppression() -> None:
    base = _synthetic_rusanov_audit(
        margin=2.0e-8,
        jump=5.0e-8,
        code=0,
    )
    switched = _synthetic_rusanov_audit(
        margin=2.0e-8,
        jump=6.0e-8,
        code=5,
    )
    result = wp10c8i._rusanov_branch_resolution(base, switched)

    assert result["passed"]
    assert result["control_changed_face_count"] == 1
    assert result["changed_faces_jump_suppressed"]


def test_rusanov_rank_one_frechet_operator_matches_scalar_solution() -> None:
    horizon = 0.2
    value = wp10c8i._rusanov_kink_frechet_output_operators(
        np.asarray([[-2.0]]),
        np.asarray([[3.0]]),
        np.asarray([[4.0]]),
        np.asarray([[5.0]]),
        horizon,
        quadrature_order=8,
    )
    expected = 3.0 * horizon * np.exp(-2.0 * horizon) * 4.0 * 5.0

    assert value.shape == (1, 1, 1)
    assert np.isclose(value[0, 0, 0], expected, rtol=2.0e-13)


def test_rusanov_kink_null_bound_sums_branch_row_norms() -> None:
    per_output, maximum = wp10c8i._rusanov_kink_null_upper_bound(
        np.asarray([[[0.0, 2.0]], [[0.0, -3.0]]]),
        np.asarray([[1.0, 0.0]]),
        np.asarray([2.0]),
        np.asarray([1.0, 1.0]),
    )

    assert np.array_equal(per_output, np.asarray([2.5]))
    assert maximum == 2.5


def test_rusanov_instantaneous_output_delta_covers_flux_and_rate(
    monkeypatch,
) -> None:
    def fake_response_stack(_arrays, _metadata, _level_index):
        return (
            np.zeros((2, 2)),
            np.ones(2),
            ("interface_1_rest_mass", "rate"),
            {
                "scientific": (0, 0),
                "full_log_h_over_r": (0, 0),
                "macro_interface_flux": (0, 1),
                "coarse_coordinate_rate": (1, 2),
            },
        )

    monkeypatch.setattr(wp10c8i, "_response_stack", fake_response_stack)
    arrays = {
        "production_rusanov_kink_physical_flux_left_factors": np.asarray(
            [[6.0], [0.0], [0.0], [0.0], [0.0]]
        ),
        "production_rusanov_kink_generator_left_factors": np.asarray(
            [[4.0], [5.0]]
        ),
        "production_rusanov_kink_generator_right_factors": np.asarray(
            [[2.0], [3.0]]
        ),
        "production_rusanov_kink_face_indices": np.asarray([3]),
        "interface_flux_scales": np.asarray([2.0]),
        "shell_edge_indices": np.asarray([0, 3]),
        "level_0_constraints": np.asarray([[1.0, 2.0]]),
    }
    metadata = {
        "interface_flux_names": ("interface_1_rest_mass",),
    }

    delta = wp10c8i._rusanov_kink_instantaneous_output_deltas(
        arrays,
        metadata,
        0,
    )

    assert delta.shape == (1, 2, 2)
    assert np.array_equal(delta[0, 0], np.asarray([6.0, 9.0]))
    assert np.array_equal(
        delta[0, 1],
        wp10c8i.COORDINATE_RATE_WINDOW_SECONDS
        * np.asarray([28.0, 42.0]),
    )
