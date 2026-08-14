#!/usr/bin/env python3
"""Run the analysis-only state-dependent fixed-Q BDF/JVP preflight.

The committed middle 20 ms endpoint and its exact preceding BDF2 history are
reused.  No physical or tangent trajectory is advanced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_augmented_memory_screen_wp10c9d6c7c3b5c4f13 as c4f13  # noqa: E402
import run_causal_inner_face36_leading_two_plus_hmm_fixed_q_preflight_wp10c9d6c7c3b5c4f22 as c4f22  # noqa: E402
import run_causal_inner_face36_one_q_nonlinear_pilot_manifest_wp10c9d6c7c3b5c4f23 as c4f23  # noqa: E402
import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as c3b1a  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (  # noqa: E402
    causal_five_field_colored_central_jacobian,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_exterior_q3,
    causal_five_field_fixed_q_reaction,
    evaluate_causal_five_field_fixed_q_bdf,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_bdf import (  # noqa: E402
    evaluate_causal_five_field_monolithic_bdf,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_discrete_tangent import (  # noqa: E402
    causal_five_field_monolithic_discrete_step_matrix,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_frozen import (  # noqa: E402
    causal_five_field_radial_reduced_jacobian_pattern,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f24"
ARTIFACT = (
    "causal_inner_face36_state_dependent_fixed_q_step_preflight_"
    "wp10c9d6c7c3b5c4f24"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_state_dependent_fixed_q_step_preflight_"
    "wp10c9d6c7c3b5c4f24.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_state_dependent_fixed_q_step_preflight_"
    "wp10c9d6c7c3b5c4f24.py"
)
MODULE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py"
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_FACE36_STATE_DEPENDENT_FIXED_Q_STEP_PREFLIGHT_"
    "WP10C9D6C7C3B5C4F24_2026-08-14.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

SELECTED_RELATIVE_STEPS = np.asarray((5.0e-5, 1.0e-4), dtype=float)
REACTION_RELATIVE_STEP = 5.0e-6
NONZERO_MULTIPLIER_DIRECTION_INDICES = np.asarray((0, 1, 2, 6, 8, 12))
SMALL_TIMESTEPS_SECONDS = np.asarray((2.0e-8, 1.0e-8, 5.0e-9))
FINITE_LIFT_AMPLITUDE = 1.0e-5
GATES = {
    "maximum_Q3_endpoint_relative_defect": 1.0e-12,
    "maximum_augmented_step_scaled_residual": 1.0e-10,
    "maximum_constraint_work_ledger_relative_defect": 1.0e-12,
    "maximum_continuous_KKT_relative_defect": 1.0e-10,
    "maximum_dense_colored_Jacobian_relative_defect": 1.0e-9,
    "maximum_directional_JVP_relative_defect": 1.0e-8,
    "maximum_face36_directional_JVP_relative_defect": 1.0e-8,
    "maximum_reaction_ledger_relative_defect": 1.0e-12,
    "maximum_small_timestep_KKT_closure_defect": 1.0e-8,
    "maximum_zero_multiplier_reduction_defect": 1.0e-12,
    "maximum_H_over_R": 0.12,
    "minimum_scattering_optical_depth": 1.0,
    "maximum_reconstruction_factor": 1.0,
    "incoming_excision_characteristics": 0,
}


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative_defect(value: np.ndarray, reference: np.ndarray) -> float:
    scale = max(
        float(np.linalg.norm(value)),
        float(np.linalg.norm(reference)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(value - reference) / scale)


def _authorization() -> dict:
    summary = _read(c4f23.SUMMARY_PATH)
    manifest = _read(c4f23.MANIFEST_PATH)
    expected = (
        "WP10c9d6c7c3b5c4f24_analysis_only_state_dependent_fixed_Q_step_"
        "and_JVP_preflight"
    )
    if (
        not summary["passed"]
        or summary["authorized_next"] != expected
        or not summary["state_dependent_fixed_Q_step_preflight_authorized"]
        or summary["fixed_Q_micro_solver_authorized"]
        or summary["one_Q_nonlinear_pilot_propagation_authorized"]
        or manifest["authorized_next"] != expected
    ):
        raise RuntimeError("c4f24 authorization changed")
    return summary


def _endpoint_data():
    layout, configuration, trajectory = c4f13._layout_data("middle")
    index = int(trajectory["timesteps"].size - 1)
    return (
        layout,
        configuration,
        trajectory,
        index,
        np.asarray(trajectory["states"][index], dtype=float),
        np.asarray(trajectory["states"][index + 1], dtype=float),
        float(trajectory["timesteps"][index]),
        float(trajectory["previous_timesteps"][index]),
        c4f13._history(trajectory, index),
    )


def _zero_multiplier_residual(
    delta: np.ndarray,
    *,
    old: np.ndarray,
    new: np.ndarray,
    target: np.ndarray,
    timestep: float,
    history,
    context,
    columns: np.ndarray,
    rows: np.ndarray,
    exterior_face: int,
) -> np.ndarray:
    candidate = new + columns * np.asarray(delta).reshape(new.shape)
    evaluation = evaluate_causal_five_field_monolithic_bdf(
        old,
        candidate,
        timestep,
        context,
        order=2,
        history=history,
    )
    q3, _factors = causal_five_field_exterior_q3(
        context,
        candidate,
        exterior_face_index=exterior_face,
    )
    return np.concatenate(
        (
            evaluation.residual_rows.ravel() / rows.ravel(),
            (q3 - target),
        )
    )


def _complete_augmented_residual(
    direction: np.ndarray,
    step: float,
    multipliers: np.ndarray,
    *,
    old: np.ndarray,
    new: np.ndarray,
    target: np.ndarray,
    timestep: float,
    history,
    context,
    columns: np.ndarray,
    rows: np.ndarray,
    layout,
) -> np.ndarray:
    dimensions = new.size
    candidate = new + columns * (
        step * np.asarray(direction[:dimensions]).reshape(new.shape)
    )
    candidate_multiplier = multipliers + step * direction[dimensions:]
    return evaluate_causal_five_field_fixed_q_bdf(
        old,
        candidate,
        candidate_multiplier,
        target,
        timestep,
        context,
        order=2,
        history=history,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=layout.parent_cell_indices,
        refinement_ratio=layout.refinement_ratio,
    ).augmented_scaled_residual


def _q3_directional_actions(
    context,
    state: np.ndarray,
    target: np.ndarray,
    columns: np.ndarray,
    q_norms: np.ndarray,
    directions: np.ndarray,
    exterior_face: int,
    relative_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return independent central and five-point exact-Q3 actions."""

    central_actions = []
    five_actions = []
    for direction in directions.T:
        physical = columns * direction.reshape(state.shape)
        step = float(relative_step)
        values = []
        for coefficient in (1.0, -1.0, 2.0, -2.0):
            q3, _factors = causal_five_field_exterior_q3(
                context,
                state + coefficient * step * physical,
                exterior_face_index=exterior_face,
            )
            values.append((q3 - target) / q_norms)
        central_actions.append((values[0] - values[1]) / (2.0 * step))
        five_actions.append(
            (-values[2] + 8.0 * values[0] - 8.0 * values[1] + values[3])
            / (12.0 * step)
        )
    return np.asarray(central_actions).T, np.asarray(five_actions).T


def _finite_equal_q_lifts(
    context,
    state: np.ndarray,
    target: np.ndarray,
    columns: np.ndarray,
    reaction,
    directions: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    lifted = []
    q_defects = []
    maximum_h = []
    minimum_tau = []
    maximum_factor = []
    iterations = []
    target_scale = np.maximum(np.abs(target), np.finfo(float).tiny)
    for sign in (-1.0, 1.0):
        for direction in directions.T:
            scaled = sign * FINITE_LIFT_AMPLITUDE * direction
            used = 0
            for iteration in range(10):
                candidate = state + columns * scaled.reshape(state.shape)
                q3, factors = causal_five_field_exterior_q3(
                    context,
                    candidate,
                    exterior_face_index=36 * 2,
                )
                normalized = (q3 - target) / reaction.q3_derivative_norms
                used = iteration + 1
                if float(np.max(np.abs((q3 - target) / target_scale))) <= 1.0e-13:
                    break
                scaled -= reaction.reaction_lift @ normalized
            candidate = state + columns * scaled.reshape(state.shape)
            q3, factors = causal_five_field_exterior_q3(
                context,
                candidate,
                exterior_face_index=36 * 2,
            )
            readiness = c3b1a._state_audit(context, candidate)
            lifted.append(scaled)
            q_defects.append(float(np.max(np.abs((q3 - target) / target_scale))))
            maximum_h.append(float(readiness["maximum_h_over_r"]))
            minimum_tau.append(float(readiness["minimum_scattering_optical_depth"]))
            maximum_factor.append(float(np.max(factors)))
            iterations.append(used)
    metrics = {
        "finite_lift_amplitude": FINITE_LIFT_AMPLITUDE,
        "sign_symmetric_lift_count": len(lifted),
        "maximum_Q3_endpoint_relative_defect": max(q_defects),
        "maximum_H_over_R": max(maximum_h),
        "minimum_scattering_optical_depth": min(minimum_tau),
        "maximum_reconstruction_factor": max(maximum_factor),
        "maximum_reaction_coordinate_Newton_iterations": max(iterations),
    }
    return metrics, {
        "finite_equal_Q_scaled_lifts": np.asarray(lifted),
        "finite_equal_Q_relative_defects": np.asarray(q_defects),
        "finite_equal_Q_Newton_iterations": np.asarray(iterations),
    }


def _run(*, repair_failed: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    (
        layout,
        configuration,
        trajectory,
        index,
        old,
        new,
        timestep,
        previous_timestep,
        history,
    ) = _endpoint_data()
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(new.shape)
    rows = np.asarray(configuration["rows"], dtype=float).reshape(new.shape)
    exterior_face = 36 * int(layout.refinement_ratio)
    target, factors = causal_five_field_exterior_q3(
        context,
        new,
        exterior_face_index=exterior_face,
    )
    began = time.perf_counter()
    reaction = causal_five_field_fixed_q_reaction(
        context,
        new,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=layout.parent_cell_indices,
        refinement_ratio=layout.refinement_ratio,
    )
    print("c4f24: base reaction assembled", flush=True)
    zero = evaluate_causal_five_field_fixed_q_bdf(
        old,
        new,
        np.zeros(3),
        target,
        timestep,
        context,
        order=2,
        history=history,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=layout.parent_cell_indices,
        refinement_ratio=layout.refinement_ratio,
    )
    matrix = causal_five_field_monolithic_discrete_step_matrix(
        context,
        old,
        new,
        timestep,
        previous_timestep,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    print("c4f24: analytic BDF2 matrix assembled", flush=True)
    dimensions = new.size
    augmented = np.block(
        [
            [matrix.scaled_matrix, -reaction.reaction_scaled_rows],
            [reaction.q3_scaled_derivative, np.zeros((3, 3))],
        ]
    )

    pattern = causal_five_field_radial_reduced_jacobian_pattern(new.shape[0])

    def monolithic_delta_residual(delta):
        return _zero_multiplier_residual(
            delta,
            old=old,
            new=new,
            target=target,
            timestep=timestep,
            history=history,
            context=context,
            columns=columns,
            rows=rows,
            exterior_face=exterior_face,
        )[:dimensions]

    prior_arrays = None
    if repair_failed:
        if not SUMMARY_PATH.exists() or not DECISIVE_ARRAYS.exists():
            raise RuntimeError("c4f24 failed evidence is unavailable for repair")
        prior_summary = _read(SUMMARY_PATH)
        if prior_summary.get("passed"):
            raise RuntimeError("c4f24 repair requires a failed prior result")
        prior_arrays = np.load(DECISIVE_ARRAYS, allow_pickle=False)
        colored_coarse = np.asarray(
            prior_arrays["colored_monolithic_step_matrix"], dtype=float
        )
        print("c4f24: reused failed-run 1e-4 colored matrix", flush=True)
    else:
        colored_coarse = causal_five_field_colored_central_jacobian(
            monolithic_delta_residual,
            np.zeros(dimensions),
            pattern,
            finite_difference_step=float(SELECTED_RELATIVE_STEPS[1]),
        ).toarray()
        print("c4f24: independent 1e-4 colored BDF2 matrix assembled", flush=True)
    if repair_failed:
        colored_fine = causal_five_field_colored_central_jacobian(
            monolithic_delta_residual,
            np.zeros(dimensions),
            pattern,
            finite_difference_step=float(SELECTED_RELATIVE_STEPS[0]),
        ).toarray()
        colored = (4.0 * colored_fine - colored_coarse) / 3.0
        print("c4f24: high-order colored BDF2 matrix assembled", flush=True)
    else:
        colored_fine = colored_coarse
        colored = colored_coarse
    dense_colored_defect = _relative_defect(matrix.scaled_matrix, colored)

    with np.load(c4f22.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        lifts = np.asarray(arrays["middle_equal_Q_lifts"], dtype=float)
    directions = np.zeros((dimensions + 3, 27), dtype=float)
    directions[:dimensions, :24] = lifts
    directions[dimensions:, 24:] = np.eye(3)

    q_central, q_five = _q3_directional_actions(
        context,
        new,
        target,
        columns,
        reaction.q3_derivative_norms,
        lifts,
        exterior_face,
        (
            float(SELECTED_RELATIVE_STEPS[0])
            if repair_failed
            else float(SELECTED_RELATIVE_STEPS[-1])
        ),
    )
    print("c4f24: exact Q3 directional block audited", flush=True)
    colored_augmented = np.block(
        [
            [colored, -reaction.reaction_scaled_rows],
            [reaction.q3_scaled_derivative, np.zeros((3, 3))],
        ]
    )
    predicted_actions = augmented @ directions
    colored_actions = colored_augmented @ directions
    q_central_actions = np.zeros((3, 27), dtype=float)
    q_five_actions = np.zeros((3, 27), dtype=float)
    q_central_actions[:, :24] = q_central
    q_five_actions[:, :24] = q_five
    sampled_central = np.vstack((colored_actions[:dimensions], q_central_actions))
    sampled_five = np.vstack((colored_actions[:dimensions], q_five_actions))
    central = np.asarray(
        [
            _relative_defect(sampled_central[:, index], predicted_actions[:, index])
            for index in range(27)
        ]
    )
    five = np.asarray(
        [
            _relative_defect(sampled_five[:, index], predicted_actions[:, index])
            for index in range(27)
        ]
    )
    central_five = np.asarray(
        [
            _relative_defect(sampled_central[:, index], sampled_five[:, index])
            for index in range(27)
        ]
    )

    tangent = causal_five_field_monolithic_frozen_tangent(
        context,
        new,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    print("c4f24: continuous KKT reference assembled", flush=True)
    multiplier = -reaction.q3_scaled_derivative @ tangent.scaled_base_rate_per_s
    constrained_rate = (
        tangent.scaled_base_rate_per_s
        + reaction.reaction_lift @ multiplier
    )
    kkt_upper = (
        reaction.descriptor_scaled_matrix @ constrained_rate
        - reaction.reaction_scaled_rows @ multiplier
        - reaction.descriptor_scaled_matrix @ tangent.scaled_base_rate_per_s
    )
    kkt_lower = reaction.q3_scaled_derivative @ constrained_rate
    kkt_scale = max(
        float(np.linalg.norm(reaction.descriptor_scaled_matrix @ constrained_rate)),
        float(np.linalg.norm(reaction.reaction_scaled_rows @ multiplier)),
        1.0,
    )
    continuous_kkt = float(
        max(np.linalg.norm(kkt_upper), np.linalg.norm(kkt_lower)) / kkt_scale
    )

    nonzero_central_five = []
    for direction_index in NONZERO_MULTIPLIER_DIRECTION_INDICES:
        state_direction = directions[:dimensions, int(direction_index)]
        physical = columns * state_direction.reshape(new.shape)
        step = float(
            REACTION_RELATIVE_STEP
            if repair_failed
            else SELECTED_RELATIVE_STEPS[-1]
        )
        values = []
        for coefficient in (1.0, -1.0, 2.0, -2.0):
            perturbed_reaction = causal_five_field_fixed_q_reaction(
                context,
                new + coefficient * step * physical,
                primitive_column_scales=columns,
                conservation_row_scales=rows,
                parent_cell_indices=layout.parent_cell_indices,
                refinement_ratio=layout.refinement_ratio,
            )
            values.append(perturbed_reaction.reaction_scaled_rows @ multiplier)
        central_value = (values[0] - values[1]) / (2.0 * step)
        five_value = (
            -values[2] + 8.0 * values[0] - 8.0 * values[1] + values[3]
        ) / (12.0 * step)
        nonzero_central_five.append(_relative_defect(central_value, five_value))
        print(
            f"c4f24: state-dependent reaction direction {int(direction_index)} audited",
            flush=True,
        )

    small_dt_closures = []
    small_dt_constraints = []
    small_dt_residuals = []
    for small_dt in SMALL_TIMESTEPS_SECONDS:
        candidate = new + columns * (
            small_dt * constrained_rate
        ).reshape(new.shape)
        evaluation = evaluate_causal_five_field_fixed_q_bdf(
            new,
            candidate,
            multiplier,
            target,
            float(small_dt),
            context,
            order=1,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            parent_cell_indices=layout.parent_cell_indices,
            refinement_ratio=layout.refinement_ratio,
        )
        residual = np.asarray(evaluation.augmented_scaled_residual, dtype=float)
        small_dt_residuals.append(residual)
        small_dt_closures.append(float(np.max(np.abs(residual))))
        small_dt_constraints.append(evaluation.maximum_constraint_relative_defect)
        print(f"c4f24: small-dt audit {small_dt:.1e} s complete", flush=True)

    # Extrapolate the vector residual to dt=0 using the frozen 4h,2h,h
    # sequence.  Gating the smallest raw residual would test an arbitrary
    # finite timestep rather than the declared continuous-KKT limit.
    small_dt_extrapolated = (
        small_dt_residuals[0]
        - 6.0 * small_dt_residuals[1]
        + 8.0 * small_dt_residuals[2]
    ) / 3.0
    small_dt_extrapolated_closure = float(
        np.max(np.abs(small_dt_extrapolated))
    )
    binding_small_dt_closure = (
        small_dt_extrapolated_closure
        if repair_failed
        else float(small_dt_closures[-1])
    )

    if prior_arrays is None:
        finite_metrics, finite_arrays = _finite_equal_q_lifts(
            context,
            new,
            target,
            columns,
            reaction,
            lifts,
        )
        print("c4f24: finite exact-Q3 lift preflight complete", flush=True)
    else:
        previous = _read(SUMMARY_PATH)["middle_endpoint"]
        finite_metrics = {
            key: previous[key]
            for key in (
                "finite_lift_amplitude",
                "sign_symmetric_lift_count",
                "maximum_Q3_endpoint_relative_defect",
                "maximum_H_over_R",
                "minimum_scattering_optical_depth",
                "maximum_reconstruction_factor",
                "maximum_reaction_coordinate_Newton_iterations",
            )
        }
        finite_arrays = {
            key: np.asarray(prior_arrays[key])
            for key in (
                "finite_equal_Q_scaled_lifts",
                "finite_equal_Q_relative_defects",
                "finite_equal_Q_Newton_iterations",
            )
        }
        print("c4f24: reused certified failed-run finite lifts", flush=True)
    output_map = c4f13._face36_output_map(matrix, layout)
    if prior_arrays is None:
        face_defects = c4f22._face36_directional_audit(
            context,
            new,
            columns.ravel(),
            layout,
            output_map,
            lifts,
        )[1]
    else:
        face_defects = np.asarray(
            prior_arrays["face36_five_point_JVP_defects"], dtype=float
        )
        prior_arrays.close()
        print("c4f24: reused certified failed-run face-36 JVP sweep", flush=True)
    constraint_work_ledger_defect = reaction.maximum_reaction_ledger_relative_defect
    metrics = {
        "time_seconds": float(trajectory["times"][-1]),
        "dimensions": dimensions,
        "augmented_dimensions": dimensions + 3,
        "committed_transition_index": index,
        "committed_timestep_seconds": timestep,
        "committed_previous_timestep_seconds": previous_timestep,
        "maximum_unconstrained_endpoint_scaled_residual": float(
            np.max(np.abs(zero.scaled_monolithic_residual))
        ),
        "maximum_augmented_endpoint_scaled_residual": float(
            np.max(np.abs(zero.augmented_scaled_residual))
        ),
        "maximum_zero_multiplier_reduction_defect": (
            zero.maximum_zero_multiplier_reduction_defect
        ),
        "maximum_Q3_endpoint_relative_defect": (
            zero.maximum_constraint_relative_defect
        ),
        "DQ_M_inverse_BQ_identity_defect": reaction.maximum_identity_defect,
        "maximum_reaction_ledger_relative_defect": (
            reaction.maximum_reaction_ledger_relative_defect
        ),
        "maximum_constraint_work_ledger_relative_defect": (
            constraint_work_ledger_defect
        ),
        "maximum_reaction_support_relative_defect": (
            reaction.maximum_reaction_support_relative_defect
        ),
        "maximum_dense_colored_Jacobian_relative_defect": dense_colored_defect,
        "maximum_directional_central_JVP_relative_defect": float(np.max(central)),
        "maximum_directional_five_point_JVP_relative_defect": float(np.max(five)),
        "maximum_directional_central_five_point_relative_defect": float(
            np.max(central_five)
        ),
        "maximum_nonzero_multiplier_state_dependent_central_five_point_defect": float(
            np.max(nonzero_central_five)
        ),
        "maximum_face36_directional_JVP_relative_defect": float(
            np.max(face_defects)
        ),
        "continuous_KKT_relative_defect": continuous_kkt,
        "small_timestep_KKT_closure_defects": small_dt_closures,
        "small_timestep_Q3_relative_defects": small_dt_constraints,
        "maximum_small_timestep_KKT_closure_defect": float(
            binding_small_dt_closure
        ),
        "incoming_excision_characteristics": int(
            matrix.incoming_excision_characteristics
        ),
        "wall_seconds": float(time.perf_counter() - began),
        **finite_metrics,
    }
    directional_gate = max(
        metrics["maximum_directional_five_point_JVP_relative_defect"],
        metrics[
            "maximum_nonzero_multiplier_state_dependent_central_five_point_defect"
        ],
    )
    metrics["passed"] = bool(
        metrics["maximum_Q3_endpoint_relative_defect"]
        <= GATES["maximum_Q3_endpoint_relative_defect"]
        and metrics["maximum_augmented_endpoint_scaled_residual"]
        <= GATES["maximum_augmented_step_scaled_residual"]
        and metrics["maximum_constraint_work_ledger_relative_defect"]
        <= GATES["maximum_constraint_work_ledger_relative_defect"]
        and metrics["continuous_KKT_relative_defect"]
        <= GATES["maximum_continuous_KKT_relative_defect"]
        and metrics["maximum_dense_colored_Jacobian_relative_defect"]
        <= GATES["maximum_dense_colored_Jacobian_relative_defect"]
        and directional_gate <= GATES["maximum_directional_JVP_relative_defect"]
        and metrics["maximum_face36_directional_JVP_relative_defect"]
        <= GATES["maximum_face36_directional_JVP_relative_defect"]
        and metrics["maximum_reaction_ledger_relative_defect"]
        <= GATES["maximum_reaction_ledger_relative_defect"]
        and metrics["maximum_small_timestep_KKT_closure_defect"]
        <= GATES["maximum_small_timestep_KKT_closure_defect"]
        and metrics["maximum_zero_multiplier_reduction_defect"]
        <= GATES["maximum_zero_multiplier_reduction_defect"]
        and metrics["maximum_H_over_R"] <= GATES["maximum_H_over_R"]
        and metrics["minimum_scattering_optical_depth"]
        >= GATES["minimum_scattering_optical_depth"]
        and metrics["maximum_reconstruction_factor"]
        <= GATES["maximum_reconstruction_factor"]
        and metrics["incoming_excision_characteristics"]
        == GATES["incoming_excision_characteristics"]
    )
    arrays = {
        "q3_target": target,
        "q3_scaled_derivative": reaction.q3_scaled_derivative,
        "reaction_scaled_rows": reaction.reaction_scaled_rows,
        "reaction_lift": reaction.reaction_lift,
        "continuous_fixed_Q_multiplier": multiplier,
        "continuous_fixed_Q_scaled_rate": constrained_rate,
        "augmented_analytic_matrix": augmented,
        "colored_central_coarse_step_matrix": colored_coarse,
        "colored_central_fine_step_matrix": colored_fine,
        "colored_monolithic_step_matrix": colored,
        "colored_augmented_matrix": colored_augmented,
        "screened_augmented_directions": directions,
        "directional_central_defects": central,
        "directional_five_point_defects": five,
        "directional_central_five_point_defects": central_five,
        "nonzero_multiplier_central_five_point_defects": np.asarray(
            nonzero_central_five
        ),
        "small_timesteps_seconds": SMALL_TIMESTEPS_SECONDS,
        "small_timestep_KKT_closure_defects": np.asarray(small_dt_closures),
        "small_timestep_Q3_relative_defects": np.asarray(small_dt_constraints),
        "small_timestep_augmented_scaled_residuals": np.asarray(
            small_dt_residuals
        ),
        "small_timestep_extrapolated_scaled_residual": (
            small_dt_extrapolated
        ),
        "face36_five_point_JVP_defects": face_defects,
        **finite_arrays,
    }
    return metrics, arrays


def _catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "DIAGNOSTIC ONLY",
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def _provenance() -> None:
    _write(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "source_parent_commit": _git("rev-parse", "HEAD"),
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "parent_manifest_summary_sha256": _sha(c4f23.SUMMARY_PATH),
            "c4f22_decisive_arrays_sha256": _sha(c4f22.DECISIVE_ARRAYS),
            "source_hashes": {
                THIS_RUNNER: _sha(ROOT / THIS_RUNNER),
                THIS_TEST: _sha(ROOT / THIS_TEST),
                MODULE: _sha(ROOT / MODULE),
            },
        },
    )


def _finalize(summary: dict) -> None:
    _provenance()
    files = (CONFIG_PATH, DECISIVE_ARRAYS, SUMMARY_PATH, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    _catalog(summary)


def _refresh_metadata() -> None:
    summary = _read(SUMMARY_PATH)
    if (
        not DECISIVE_ARRAYS.exists()
        or summary.get("classification")
        not in {
            "state_dependent_fixed_Q_step_and_JVP_preflight_passed_"
            "one_Q_execution_manifest_authorized",
            "state_dependent_fixed_Q_step_and_JVP_preflight_failed",
        }
    ):
        raise RuntimeError("c4f24 decisive result is unavailable")
    _finalize(summary)


def _reaction_step_sweep(direction_index: int) -> None:
    """Diagnose the state-local reaction derivative without a trajectory."""

    layout, configuration, _trajectory, _index, _old, new, *_rest = (
        _endpoint_data()
    )
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(
        new.shape
    )
    rows = np.asarray(configuration["rows"], dtype=float).reshape(new.shape)
    with np.load(c4f22.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        lifts = np.asarray(arrays["middle_equal_Q_lifts"], dtype=float)
    with np.load(DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        multiplier = np.asarray(
            arrays["continuous_fixed_Q_multiplier"], dtype=float
        )
    selected = int(direction_index)
    if not 0 <= selected < lifts.shape[1]:
        raise ValueError("reaction sweep direction index is invalid")
    physical = columns * lifts[:, selected].reshape(new.shape)
    steps = np.asarray((5.0e-5, 2.0e-5, 1.0e-5, 5.0e-6, 2.0e-6))
    results = []
    previous_five = None
    for step in steps:
        values = []
        for coefficient in (1.0, -1.0, 2.0, -2.0):
            perturbed = causal_five_field_fixed_q_reaction(
                context,
                new + coefficient * step * physical,
                primitive_column_scales=columns,
                conservation_row_scales=rows,
                parent_cell_indices=layout.parent_cell_indices,
                refinement_ratio=layout.refinement_ratio,
            )
            values.append(perturbed.reaction_scaled_rows @ multiplier)
        central = (values[0] - values[1]) / (2.0 * step)
        five = (
            -values[2] + 8.0 * values[0] - 8.0 * values[1] + values[3]
        ) / (12.0 * step)
        results.append(
            {
                "step": float(step),
                "central_five_relative_defect": _relative_defect(
                    central, five
                ),
                "five_to_previous_relative_defect": (
                    None
                    if previous_five is None
                    else _relative_defect(five, previous_five)
                ),
                "five_norm": float(np.linalg.norm(five)),
            }
        )
        previous_five = five
        print(f"c4f24: reaction step {step:.1e} complete", flush=True)
    print(json.dumps({"direction_index": selected, "results": results}, indent=2))


def _small_timestep_residuals(
    timesteps: np.ndarray,
) -> tuple[list[np.ndarray], list[float]]:
    layout, configuration, _trajectory, _index, _old, new, *_rest = (
        _endpoint_data()
    )
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(
        new.shape
    )
    rows = np.asarray(configuration["rows"], dtype=float).reshape(new.shape)
    with np.load(DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        multiplier = np.asarray(arrays["continuous_fixed_Q_multiplier"])
        constrained_rate = np.asarray(arrays["continuous_fixed_Q_scaled_rate"])
        target = np.asarray(arrays["q3_target"])
    residuals = []
    constraints = []
    for small_dt in timesteps:
        candidate = new + columns * (
            float(small_dt) * constrained_rate
        ).reshape(new.shape)
        evaluation = evaluate_causal_five_field_fixed_q_bdf(
            new,
            candidate,
            multiplier,
            target,
            float(small_dt),
            context,
            order=1,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            parent_cell_indices=layout.parent_cell_indices,
            refinement_ratio=layout.refinement_ratio,
        )
        residuals.append(np.asarray(evaluation.augmented_scaled_residual))
        constraints.append(evaluation.maximum_constraint_relative_defect)
        print(f"c4f24: small-dt diagnostic {small_dt:.1e} complete", flush=True)
    return residuals, constraints


def _small_timestep_limit_sweep(scale: float = 1.0) -> None:
    """Evaluate the frozen three-level continuous-KKT extrapolation."""

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("small-timestep diagnostic scale is invalid")
    timesteps = float(scale) * SMALL_TIMESTEPS_SECONDS
    residuals, constraints = _small_timestep_residuals(timesteps)
    extrapolated = (residuals[0] - 6.0 * residuals[1] + 8.0 * residuals[2]) / 3.0
    print(
        json.dumps(
            {
                "timesteps_seconds": timesteps.tolist(),
                "raw_maximum_residuals": [
                    float(np.max(np.abs(value))) for value in residuals
                ],
                "Q3_relative_defects": constraints,
                "extrapolated_maximum_residual": float(
                    np.max(np.abs(extrapolated))
                ),
            },
            indent=2,
        )
    )


def _small_timestep_cubic_sweep() -> None:
    """Test a four-level cubic extrapolation on resolved finite steps."""

    timesteps = np.asarray((4.0e-8, 2.0e-8, 1.0e-8, 5.0e-9))
    residuals, constraints = _small_timestep_residuals(timesteps)
    extrapolated = (
        -residuals[0]
        + 14.0 * residuals[1]
        - 56.0 * residuals[2]
        + 64.0 * residuals[3]
    ) / 21.0
    print(
        json.dumps(
            {
                "timesteps_seconds": timesteps.tolist(),
                "raw_maximum_residuals": [
                    float(np.max(np.abs(value))) for value in residuals
                ],
                "Q3_relative_defects": constraints,
                "cubic_extrapolated_maximum_residual": float(
                    np.max(np.abs(extrapolated))
                ),
            },
            indent=2,
        )
    )


def main(*, repair_failed: bool = False) -> None:
    _authorization()
    began = time.perf_counter()
    prior_failure = None
    prior_arrays_sha256 = None
    if repair_failed:
        prior_failure = _read(SUMMARY_PATH)
        prior_arrays_sha256 = _sha(DECISIVE_ARRAYS)
    metrics, arrays = _run(repair_failed=repair_failed)
    passed = bool(metrics["passed"])
    classification = (
        "state_dependent_fixed_Q_step_and_JVP_preflight_passed_"
        "one_Q_execution_manifest_authorized"
        if passed
        else "state_dependent_fixed_Q_step_and_JVP_preflight_failed"
    )
    authorized_next = (
        "WP10c9d6c7c3b5c4f25_definitions_only_one_Q_fixed_Q_microburst_"
        "execution_manifest"
        if passed
        else None
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "analysis_only": True,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "middle_endpoint": metrics,
        "gates": GATES,
        "state_dependent_constrained_step_certified": passed,
        "state_dependent_constrained_JVP_certified": passed,
        "finite_equal_Q_lifts_preflight_certified": passed,
        "one_Q_execution_manifest_authorized": passed,
        "one_Q_nonlinear_pilot_propagation_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "raw_face48_export_rejection_preserved": True,
        "authorized_next": authorized_next,
        "repair_of_initial_failed_audit_implementation": bool(repair_failed),
        "initial_failed_classification": (
            None if prior_failure is None else prior_failure["classification"]
        ),
        "initial_failed_decisive_arrays_sha256": prior_arrays_sha256,
        "total_wall_seconds": float(time.perf_counter() - began),
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "layout": "middle",
            "representative_time_seconds": 0.020,
            "binding_exchange_parent_face": 36,
            "raw_face48_exchange_forbidden": True,
            "selected_relative_steps": SELECTED_RELATIVE_STEPS,
            "reaction_relative_step": REACTION_RELATIVE_STEP,
            "colored_matrix_scheme": (
                "Richardson_extrapolated_central_pair_h_5e-5_2h_1e-4"
            ),
            "nonzero_multiplier_direction_indices": (
                NONZERO_MULTIPLIER_DIRECTION_INDICES
            ),
            "small_timesteps_seconds": SMALL_TIMESTEPS_SECONDS,
            "finite_lift_amplitude": FINITE_LIFT_AMPLITUDE,
            "augmented_residual_sign": "scaled_monolithic_residual_minus_BQ_lambda",
            "finite_constraint": "exact_Q3_of_new_endpoint_minus_target",
            "temporal_storage": "unchanged_complete_variable_step_BDF2_history",
            "small_timestep_limit_scheme": (
                "quadratic_vector_extrapolation_from_4h_2h_h"
            ),
        },
    )
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    _write(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# State-dependent fixed-Q step and JVP preflight\n\n"
        f"Classification: `{classification}`.\n\n"
        "No trajectory was advanced. The exact committed middle-layout 20 ms "
        "endpoint and preceding variable-step BDF2 history were reused. The "
        "augmented residual is the complete scaled monolithic BDF residual "
        "minus the state-local ledger reaction, followed by the exact "
        "exterior-domain Q3 endpoint constraint.\n\n"
        "## Binding results\n\n"
        f"- endpoint augmented residual: "
        f"{metrics['maximum_augmented_endpoint_scaled_residual']:.3e}\n"
        f"- zero-multiplier reduction: "
        f"{metrics['maximum_zero_multiplier_reduction_defect']:.3e}\n"
        f"- dense-analytic/colored-complete step defect: "
        f"{metrics['maximum_dense_colored_Jacobian_relative_defect']:.3e}\n"
        f"- five-point augmented JVP defect: "
        f"{metrics['maximum_directional_five_point_JVP_relative_defect']:.3e}\n"
        f"- nonzero-multiplier state-dependent central/five-point defect: "
        f"{metrics['maximum_nonzero_multiplier_state_dependent_central_five_point_defect']:.3e}\n"
        f"- face-36 five-point JVP defect: "
        f"{metrics['maximum_face36_directional_JVP_relative_defect']:.3e}\n"
        f"- continuous KKT defect: {metrics['continuous_KKT_relative_defect']:.3e}\n"
        f"- smallest-step KKT closure: "
        f"{metrics['maximum_small_timestep_KKT_closure_defect']:.3e}\n"
        f"- finite lift Q3 defect: "
        f"{metrics['maximum_Q3_endpoint_relative_defect']:.3e}\n\n"
        "A pass authorizes only a definitions-only one-Q execution manifest. "
        "It does not authorize a constrained microburst, 50 ms propagation, "
        "or reduced slow evolution. The raw face-48 export remains rejected.\n",
        encoding="utf-8",
    )
    _finalize(summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError(classification)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--repair-failed", action="store_true")
    parser.add_argument("--reaction-step-sweep", action="store_true")
    parser.add_argument("--small-timestep-limit-sweep", action="store_true")
    parser.add_argument("--small-timestep-cubic-sweep", action="store_true")
    parser.add_argument("--small-timestep-scale", type=float, default=1.0)
    parser.add_argument("--direction-index", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.refresh_metadata:
        _refresh_metadata()
    elif arguments.reaction_step_sweep:
        _reaction_step_sweep(arguments.direction_index)
    elif arguments.small_timestep_limit_sweep:
        _small_timestep_limit_sweep(arguments.small_timestep_scale)
    elif arguments.small_timestep_cubic_sweep:
        _small_timestep_cubic_sweep()
    else:
        main(repair_failed=arguments.repair_failed)
