#!/usr/bin/env python3
"""Run the fail-fast authentic fixed-Q BDF1-to-BDF2 history ladder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_augmented_memory_screen_wp10c9d6c7c3b5c4f13 as c4f13  # noqa: E402
import run_causal_inner_face36_state_dependent_fixed_q_step_preflight_wp10c9d6c7c3b5c4f24 as c4f24  # noqa: E402
from run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a import (  # noqa: E402
    _state_audit,
)

from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    CausalFiveFieldFixedQBackwardEulerResult,
    causal_five_field_fixed_q_bdf_restart,
    causal_five_field_fixed_q_bdf_restarts_equal,
    causal_five_field_fixed_q_reaction,
    load_causal_five_field_fixed_q_bdf_restart,
    save_causal_five_field_fixed_q_bdf_restart,
    solve_causal_five_field_fixed_q_bdf,
    solve_causal_five_field_fixed_q_backward_euler,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e1"
ARTIFACT = (
    "causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1"
)
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
IMPLEMENTATION_ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_fixed_q_history_implementation_preflight_"
    "wp10c9d6c7c3b5c4f24e0"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1.py"
)
TIMESTEPS = (1.0e-7, 5.0e-8, 2.5e-8)
CASE_ORDER = (
    "primary_coarse",
    "heldout_coarse",
    "primary_middle",
    "heldout_middle",
    "primary_fine",
    "heldout_fine",
)
CASE_DEFINITIONS = {
    "primary_coarse": ("primary_20ms", 0),
    "heldout_coarse": ("heldout_16ms", 0),
    "primary_middle": ("primary_20ms", 1),
    "heldout_middle": ("heldout_16ms", 1),
    "primary_fine": ("primary_20ms", 2),
    "heldout_fine": ("heldout_16ms", 2),
}
GATES = {
    "maximum_scaled_residual": 1.0e-10,
    "maximum_Q3_relative_defect": 1.0e-12,
    "maximum_ledger_relative_defect": 1.0e-12,
    "maximum_storage_parity_relative_defect": 1.0e-9,
    "minimum_path_reconstruction_factor": 1.0 - 1.0e-12,
    "maximum_path_reconstruction_factor": 1.0 + 1.0e-12,
    "maximum_raw_Schur_condition_number": 1.0e8,
    "maximum_H_over_R": 0.12,
    "minimum_scattering_optical_depth": 1.0,
    "maximum_scaled_primitive_change": 5.0e-3,
    "maximum_complete_Jacobian_assemblies": 1,
    "minimum_state_rate_convergence_order": 0.9,
    "minimum_reaction_action_convergence_order": 0.9,
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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _identity() -> dict:
    return {
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "runner_sha256": _sha(ROOT / THIS_RUNNER),
        "implementation_summary_sha256": _sha(
            IMPLEMENTATION_ARTIFACT / "summary.json"
        ),
        "fixed_q_source_sha256": _sha(
            ROOT
            / "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py"
        ),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def _relative(value: np.ndarray, reference: np.ndarray) -> float:
    left = np.asarray(value, dtype=float)
    right = np.asarray(reference, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


def _state_data(label: str) -> dict:
    if label == "primary_20ms":
        (
            layout,
            configuration,
            trajectory,
            _index,
            _old,
            state,
            _timestep,
            _previous_timestep,
            _history,
        ) = c4f24._endpoint_data()
        expected_time = 2.0e-2
    elif label == "heldout_16ms":
        layout, configuration, trajectory = c4f13._layout_data("middle")
        matches = np.flatnonzero(
            np.asarray(trajectory["times"], dtype=float) == 1.6e-2
        )
        if matches.size != 1:
            raise RuntimeError("unique committed 16 ms state is unavailable")
        state = np.asarray(trajectory["states"][int(matches[0])], dtype=float)
        expected_time = 1.6e-2
    else:
        raise ValueError("fixed-Q state label is invalid")
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(
        state.shape
    )
    rows = np.asarray(configuration["rows"], dtype=float).reshape(state.shape)
    reaction = causal_five_field_fixed_q_reaction(
        context,
        state,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=layout.parent_cell_indices,
        refinement_ratio=layout.refinement_ratio,
        maximum_schur_condition_number=(
            GATES["maximum_raw_Schur_condition_number"]
        ),
    )
    tangent = causal_five_field_monolithic_frozen_tangent(
        context,
        state,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    multiplier = (
        -reaction.q3_scaled_derivative @ tangent.scaled_base_rate_per_s
    )
    rate = tangent.scaled_base_rate_per_s + (
        reaction.reaction_lift @ multiplier
    )
    action = reaction.reaction_lift @ multiplier
    face = 36 * int(layout.refinement_ratio)
    fields = np.asarray((0, 2, 3), dtype=int)
    output_map = np.asarray(
        tangent.spatial_tangent.shared_face_flux_scaled_jacobians,
        dtype=float,
    )[face, fields]
    return {
        "label": label,
        "time_seconds": expected_time,
        "layout": layout,
        "context": context,
        "state": np.asarray(state, dtype=float),
        "columns": columns,
        "rows": rows,
        "reaction": reaction,
        "tangent": tangent,
        "continuous_multiplier": multiplier,
        "continuous_rate": rate,
        "continuous_reaction_action": action,
        "output_map": output_map,
        "continuous_face36_rate": output_map @ rate,
    }


def _old_direct_seed(label: str, timestep_index: int, data: dict):
    if label != "primary_20ms":
        return None, None
    directory = (
        ROOT
        / "outputs/checkpoints"
        / "causal_inner_face36_fixed_q_jacobian_repair_"
        "wp10c9d6c7c3b5c4f24b"
    )
    path = directory / f"exact_step_{int(timestep_index)}.npz"
    if not path.exists():
        return None, None
    with np.load(path, allow_pickle=False) as source:
        charts = np.asarray(source["primitive_charts"], dtype=float)
        multiplier = np.asarray(source["multipliers"], dtype=float)
    if charts.shape != data["state"].shape or multiplier.shape != (3,):
        return None, None
    increment = ((charts - data["state"]) / data["columns"]).ravel()
    return increment, multiplier


def _result_metrics(
    result: CausalFiveFieldFixedQBackwardEulerResult,
    data: dict,
) -> dict:
    return {
        "accepted": result.accepted,
        "message": result.message,
        "failure_reasons": list(result.acceptance.failure_reasons),
        "iterations": result.iterations,
        "function_evaluations": result.function_evaluations,
        "exact_Jacobian_assemblies": result.exact_jacobian_assemblies,
        "Broyden_updates": result.broyden_updates,
        "linear_solves": result.linear_solves,
        "maximum_scaled_residual": result.maximum_scaled_residual,
        "maximum_Q3_relative_defect": (
            result.evaluation.maximum_constraint_relative_defect
        ),
        "maximum_storage_parity_relative_defect": (
            result.maximum_direct_rate_increment_parity_defect
        ),
        "minimum_path_reconstruction_factor": (
            result.minimum_path_reconstruction_factor
        ),
        "maximum_path_reconstruction_factor": (
            result.maximum_path_reconstruction_factor
        ),
        "maximum_reaction_channel_ledger_relative_defect": (
            result.evaluation.reaction.maximum_reaction_ledger_relative_defect
        ),
        "maximum_reaction_action_ledger_relative_defect": (
            result.maximum_multiplier_weighted_action_ledger_relative_defect
        ),
        "raw_Schur_rank": result.evaluation.reaction.raw_schur_numerical_rank,
        "raw_Schur_condition_number": (
            result.evaluation.reaction.raw_schur_condition_number
        ),
        "raw_Schur_singular_values": (
            result.evaluation.reaction.raw_schur_singular_values
        ),
        "maximum_raw_Schur_solve_relative_defect": (
            result.evaluation.reaction.maximum_raw_schur_solve_relative_defect
        ),
        "incoming_excision_characteristics": (
            result.evaluation.monolithic_evaluation
            .incoming_excision_characteristics
        ),
        "maximum_H_over_R": result.maximum_h_over_r,
        "minimum_scattering_optical_depth": (
            result.minimum_scattering_optical_depth
        ),
        "maximum_scaled_primitive_change": (
            result.maximum_scaled_primitive_change
        ),
        "maximum_scaled_Q3_rate_tangency_defect": (
            result.maximum_scaled_q3_rate_tangency_defect
        ),
        "state_rate_relative_defect": _relative(
            result.scaled_rate_per_s,
            data["continuous_rate"],
        ),
        "reaction_action_relative_defect": _relative(
            result.scaled_reaction_rate_action_per_s,
            data["continuous_reaction_action"],
        ),
        "face36_rate_relative_defect": _relative(
            data["output_map"] @ result.scaled_rate_per_s,
            data["continuous_face36_rate"],
        ),
        "acceptance": asdict(result.acceptance),
    }


def _save_result(path: Path, result, metrics: dict) -> None:
    _write_npz(
        path,
        primitive_charts=result.primitive_charts,
        primitive_increment=result.primitive_increment,
        scaled_rate_per_s=result.scaled_rate_per_s,
        scaled_interval_rate_per_s=result.scaled_interval_rate_per_s,
        multipliers=result.multipliers,
        scaled_reaction_rate_action_per_s=(
            result.scaled_reaction_rate_action_per_s
        ),
        augmented_scaled_residual=(
            result.evaluation.augmented_scaled_residual
        ),
        metrics_json=np.asarray(json.dumps(_plain(metrics), sort_keys=True)),
    )


def _bitwise_results_equal(left, right) -> bool:
    arrays = (
        (left.primitive_charts, right.primitive_charts),
        (left.primitive_increment, right.primitive_increment),
        (left.scaled_rate_per_s, right.scaled_rate_per_s),
        (left.scaled_interval_rate_per_s, right.scaled_interval_rate_per_s),
        (left.multipliers, right.multipliers),
        (
            left.scaled_reaction_rate_action_per_s,
            right.scaled_reaction_rate_action_per_s,
        ),
        (
            left.evaluation.augmented_scaled_residual,
            right.evaluation.augmented_scaled_residual,
        ),
    )
    return bool(
        all(np.array_equal(first, second) for first, second in arrays)
        and left.acceptance == right.acceptance
        and left.iterations == right.iterations
        and left.function_evaluations == right.function_evaluations
        and left.exact_jacobian_assemblies == right.exact_jacobian_assemblies
        and left.broyden_updates == right.broyden_updates
        and left.linear_solves == right.linear_solves
    )


def _solve_case(case: str) -> dict:
    if case not in CASE_DEFINITIONS:
        raise ValueError("fixed-Q ladder case is invalid")
    position = CASE_ORDER.index(case)
    for prior in CASE_ORDER[:position]:
        path = CHECKPOINT_DIRECTORY / f"{prior}.json"
        if not path.exists() or not json.loads(path.read_text())["passed"]:
            raise RuntimeError(f"prior fixed-Q ladder case {prior} did not pass")
    identity = _identity()
    identity_path = CHECKPOINT_DIRECTORY / "execution_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != _plain(identity):
            raise RuntimeError("fixed-Q ladder execution identity changed")
    else:
        _write_json(identity_path, identity)
    state_label, timestep_index = CASE_DEFINITIONS[case]
    timestep = float(TIMESTEPS[timestep_index])
    print(f"f24e1 {case}: assemble committed state", flush=True)
    data = _state_data(state_label)
    initial_increment, initial_multiplier = _old_direct_seed(
        state_label,
        timestep_index,
        data,
    )
    if initial_multiplier is None:
        initial_multiplier = data["continuous_multiplier"]
    top_left = (
        data["reaction"].descriptor_scaled_matrix / timestep
        + data["tangent"].evolving_scaled_jacobian
    )
    began = time.perf_counter()
    bdf1 = solve_causal_five_field_fixed_q_backward_euler(
        data["context"],
        data["state"],
        timestep,
        data["continuous_rate"],
        initial_multiplier,
        top_left,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        q3_target=data["reaction"].q3_value,
        constraint_row_scales=data["reaction"].q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=data["reaction"].raw_schur_inverse,
        residual_tolerance=GATES["maximum_scaled_residual"],
        constraint_tolerance=GATES["maximum_Q3_relative_defect"],
        ledger_tolerance=GATES["maximum_ledger_relative_defect"],
        storage_parity_tolerance=(
            GATES["maximum_storage_parity_relative_defect"]
        ),
        minimum_reconstruction_factor=(
            GATES["minimum_path_reconstruction_factor"]
        ),
        maximum_schur_condition_number=(
            GATES["maximum_raw_Schur_condition_number"]
        ),
        maximum_scaled_primitive_change=(
            GATES["maximum_scaled_primitive_change"]
        ),
        maximum_newton_iterations=8,
        maximum_line_search_iterations=12,
        refresh_exact_jacobian=True,
        maximum_exact_jacobian_refreshes=1,
        initial_scaled_increment=initial_increment,
        base_reaction=data["reaction"],
        physical_state_audit=_state_audit,
        require_physical_state_audit=True,
        maximum_h_over_r=GATES["maximum_H_over_R"],
        minimum_scattering_optical_depth=(
            GATES["minimum_scattering_optical_depth"]
        ),
        progress_callback=lambda payload: print(
            f"f24e1 {case} BDF1: {payload}", flush=True
        ),
    )
    bdf1_metrics = _result_metrics(bdf1, data)
    bdf1_metrics["wall_seconds"] = time.perf_counter() - began
    _save_result(CHECKPOINT_DIRECTORY / f"{case}_bdf1.npz", bdf1, bdf1_metrics)
    if not bdf1.accepted:
        metrics = {
            "case": case,
            "state_label": state_label,
            "timestep_seconds": timestep,
            "passed": False,
            "failed_stage": "BDF1",
            "BDF1": bdf1_metrics,
        }
        _write_json(CHECKPOINT_DIRECTORY / f"{case}.json", metrics)
        return metrics

    restart = causal_five_field_fixed_q_bdf_restart(
        bdf1,
        data["context"],
        data["state"],
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        maximum_schur_condition_number=(
            GATES["maximum_raw_Schur_condition_number"]
        ),
        q3_target=data["reaction"].q3_value,
        constraint_row_scales=data["reaction"].q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        elapsed_time_seconds=data["time_seconds"] + timestep,
        completed_steps=1,
        provenance=identity,
    )
    restart_path = CHECKPOINT_DIRECTORY / f"{case}_restart.npz"
    save_causal_five_field_fixed_q_bdf_restart(
        restart_path,
        data["context"],
        restart,
    )
    loaded = load_causal_five_field_fixed_q_bdf_restart(
        restart_path,
        data["context"],
        expected_provenance=identity,
    )
    restart_bitwise = causal_five_field_fixed_q_bdf_restarts_equal(
        restart,
        loaded,
    )
    endpoint_reaction = causal_five_field_fixed_q_reaction(
        data["context"],
        loaded.primitive_charts,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        maximum_schur_condition_number=(
            GATES["maximum_raw_Schur_condition_number"]
        ),
    )
    bdf2_top_left = (
        1.5 * endpoint_reaction.descriptor_scaled_matrix / timestep
        + data["tangent"].evolving_scaled_jacobian
    )

    def run_bdf2(tag: str):
        print(f"f24e1 {case} {tag}: start", flush=True)
        return solve_causal_five_field_fixed_q_bdf(
            data["context"],
            loaded.primitive_charts,
            timestep,
            bdf1.scaled_rate_per_s,
            loaded.multiplier_predictor,
            bdf2_top_left,
            order=2,
            history=loaded.history,
            primitive_column_scales=data["columns"],
            conservation_row_scales=data["rows"],
            parent_cell_indices=data["layout"].parent_cell_indices,
            refinement_ratio=data["layout"].refinement_ratio,
            q3_target=loaded.q3_target,
            constraint_row_scales=loaded.constraint_row_scales,
            reaction_channel_basis="frozen_normalized",
            reaction_channel_transform=loaded.reaction_channel_transform,
            residual_tolerance=GATES["maximum_scaled_residual"],
            constraint_tolerance=GATES["maximum_Q3_relative_defect"],
            ledger_tolerance=GATES["maximum_ledger_relative_defect"],
            storage_parity_tolerance=(
                GATES["maximum_storage_parity_relative_defect"]
            ),
            minimum_reconstruction_factor=(
                GATES["minimum_path_reconstruction_factor"]
            ),
            maximum_schur_condition_number=(
                GATES["maximum_raw_Schur_condition_number"]
            ),
            maximum_scaled_primitive_change=(
                GATES["maximum_scaled_primitive_change"]
            ),
            maximum_newton_iterations=8,
            maximum_line_search_iterations=12,
            refresh_exact_jacobian=True,
            maximum_exact_jacobian_refreshes=1,
            base_reaction=endpoint_reaction,
            physical_state_audit=_state_audit,
            require_physical_state_audit=True,
            maximum_h_over_r=GATES["maximum_H_over_R"],
            minimum_scattering_optical_depth=(
                GATES["minimum_scattering_optical_depth"]
            ),
            progress_callback=lambda payload: print(
                f"f24e1 {case} {tag}: {payload}", flush=True
            ),
        )

    began_bdf2 = time.perf_counter()
    bdf2 = run_bdf2("BDF2")
    bdf2_metrics = _result_metrics(bdf2, data)
    bdf2_metrics["wall_seconds"] = time.perf_counter() - began_bdf2
    _save_result(CHECKPOINT_DIRECTORY / f"{case}_bdf2.npz", bdf2, bdf2_metrics)
    if not bdf2.accepted:
        metrics = {
            "case": case,
            "state_label": state_label,
            "timestep_seconds": timestep,
            "passed": False,
            "failed_stage": "BDF2",
            "restart_roundtrip_bitwise": restart_bitwise,
            "BDF1": bdf1_metrics,
            "BDF2": bdf2_metrics,
        }
        _write_json(CHECKPOINT_DIRECTORY / f"{case}.json", metrics)
        return metrics
    replay = run_bdf2("replay")
    replay_bitwise = _bitwise_results_equal(bdf2, replay)
    counts_passed = bool(
        bdf1.exact_jacobian_assemblies
        <= GATES["maximum_complete_Jacobian_assemblies"]
        and bdf2.exact_jacobian_assemblies
        <= GATES["maximum_complete_Jacobian_assemblies"]
        and replay.exact_jacobian_assemblies
        <= GATES["maximum_complete_Jacobian_assemblies"]
    )
    passed = bool(
        bdf1.accepted
        and bdf2.accepted
        and replay.accepted
        and restart_bitwise
        and replay_bitwise
        and counts_passed
    )
    metrics = {
        "case": case,
        "state_label": state_label,
        "state_time_seconds": data["time_seconds"],
        "timestep_seconds": timestep,
        "passed": passed,
        "failed_stage": None if passed else "replay_or_solver_budget",
        "restart_roundtrip_bitwise": restart_bitwise,
        "BDF2_replay_bitwise": replay_bitwise,
        "Jacobian_assembly_budget_passed": counts_passed,
        "BDF1": bdf1_metrics,
        "BDF2": bdf2_metrics,
    }
    _write_json(CHECKPOINT_DIRECTORY / f"{case}.json", metrics)
    return metrics


def _order(coarse: float, fine: float) -> float:
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return float(math.log(coarse / fine) / math.log(2.0))


def _finalize() -> dict:
    cases = {}
    for case in CASE_ORDER:
        path = CHECKPOINT_DIRECTORY / f"{case}.json"
        if not path.exists():
            raise RuntimeError(f"fixed-Q ladder case {case} is missing")
        cases[case] = json.loads(path.read_text(encoding="utf-8"))
    if not all(item["passed"] for item in cases.values()):
        first = next(case for case in CASE_ORDER if not cases[case]["passed"])
        summary = {
            "schema_version": 1,
            "work_package": WORK_PACKAGE,
            "classification": "authentic_fixed_Q_history_ladder_failed",
            "passed": False,
            "first_failed_case": first,
            "cases": cases,
            "one_Q_execution_manifest_authorized": False,
            "fixed_Q_micro_solver_authorized": False,
            "reduced_slow_evolution_authorized": False,
        }
        _write_json(CHECKPOINT_DIRECTORY / "summary.json", summary)
        return summary
    state_orders = {}
    for state_label, prefix in (
        ("primary_20ms", "primary"),
        ("heldout_16ms", "heldout"),
    ):
        state_cases = [cases[f"{prefix}_{level}"] for level in (
            "coarse", "middle", "fine"
        )]
        rate_errors = [
            item["BDF2"]["state_rate_relative_defect"]
            for item in state_cases
        ]
        action_errors = [
            item["BDF2"]["reaction_action_relative_defect"]
            for item in state_cases
        ]
        state_orders[state_label] = {
            "state_rate_relative_defects": rate_errors,
            "reaction_action_relative_defects": action_errors,
            "state_rate_orders": [
                _order(rate_errors[0], rate_errors[1]),
                _order(rate_errors[1], rate_errors[2]),
            ],
            "reaction_action_orders": [
                _order(action_errors[0], action_errors[1]),
                _order(action_errors[1], action_errors[2]),
            ],
        }
    orders_passed = all(
        min(values["state_rate_orders"])
        >= GATES["minimum_state_rate_convergence_order"]
        and min(values["reaction_action_orders"])
        >= GATES["minimum_reaction_action_convergence_order"]
        for values in state_orders.values()
    )
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": (
            "authentic_fixed_Q_BDF1_BDF2_history_ladder_certified_"
            "one_Q_manifest_authorized"
            if orders_passed
            else "authentic_fixed_Q_history_ladder_order_failed"
        ),
        "passed": orders_passed,
        "cases": cases,
        "convergence": state_orders,
        "one_Q_execution_manifest_authorized": orders_passed,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    _write_json(CHECKPOINT_DIRECTORY / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASE_ORDER)
    parser.add_argument("--finalize", action="store_true")
    arguments = parser.parse_args()
    if (arguments.case is None) == (not arguments.finalize):
        raise SystemExit("select exactly one --case or --finalize")
    payload = _finalize() if arguments.finalize else _solve_case(arguments.case)
    print(json.dumps(_plain(payload), indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
