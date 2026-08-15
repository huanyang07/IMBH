#!/usr/bin/env python3
"""Certify the repaired fixed-Q Jacobian at 16 ms and in constrained BDF2.

No trajectory is advanced.  The runner reuses the committed middle-layout
trajectory, audits one existing BDF2 endpoint at 16 ms, and solves three
independent tiny constrained BDF2 steps with exact equal-Q3 histories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_augmented_memory_screen_wp10c9d6c7c3b5c4f13 as c4f13  # noqa: E402
import run_causal_inner_face36_fixed_q_jacobian_repair_wp10c9d6c7c3b5c4f24b as c4f24b  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    CausalFiveFieldFixedQReaction,
    causal_five_field_exterior_q3,
    causal_five_field_fixed_q_augmented_step_matrix,
    causal_five_field_fixed_q_reaction,
    evaluate_causal_five_field_fixed_q_bdf,
    solve_causal_five_field_fixed_q_bdf,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_bdf import (  # noqa: E402
    CausalFiveFieldMonolithicBDFHistory,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_dae import (  # noqa: E402
    causal_five_field_monolithic_storage_increment,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
)


ARTIFACT = (
    "causal_inner_face36_fixed_q_second_state_bdf2_preflight_"
    "wp10c9d6c7c3b5c4f24c"
)
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
BASE_REFERENCE_PATH = CHECKPOINT_DIRECTORY / "base_reference.npz"
DERIVATIVE_PATH = CHECKPOINT_DIRECTORY / "second_state_derivative.npz"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_FACE36_FIXED_Q_SECOND_STATE_BDF2_PREFLIGHT_"
    "WP10C9D6C7C3B5C4F24C_2026-08-14.md"
)
SECOND_STATE_TIME_SECONDS = 1.6e-2
STEP_TIMESTEPS_SECONDS = np.asarray(
    (2.0e-9, 1.0e-9, 5.0e-10, 8.0e-9, 4.0e-9)
)
BINDING_STEP_INDICES = (3, 4, 0)
GATES = {
    "maximum_direct_monolithic_JVP_relative_defect": 1.0e-8,
    "maximum_direct_augmented_JVP_relative_defect": 1.0e-8,
    "maximum_direct_raw_reaction_JVP_relative_defect": 1.0e-8,
    "maximum_history_Q3_relative_defect": 1.0e-12,
    "maximum_exact_step_scaled_residual": 1.0e-10,
    "maximum_exact_step_Q3_relative_defect": 1.0e-12,
    "minimum_rate_convergence_order": 0.9,
    "minimum_multiplier_convergence_order": 0.9,
}


def _write_json(path: Path, payload) -> None:
    c4f24b._write_json(path, payload)


def _write_npz(path: Path, **arrays) -> None:
    c4f24b._write_npz(path, **arrays)


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


def _relative(value: np.ndarray, reference: np.ndarray) -> float:
    scale = max(
        float(np.linalg.norm(value)),
        float(np.linalg.norm(reference)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(value - reference) / scale)


def _second_state_endpoint():
    layout, configuration, trajectory = c4f13._layout_data("middle")
    times = np.asarray(trajectory["times"], dtype=float)
    matches = np.flatnonzero(times == SECOND_STATE_TIME_SECONDS)
    if matches.size != 1 or int(matches[0]) < 1:
        raise RuntimeError("unique committed 16 ms state is unavailable")
    state_index = int(matches[0])
    step_index = state_index - 1
    return (
        layout,
        configuration,
        trajectory,
        step_index,
        np.asarray(trajectory["states"][step_index], dtype=float),
        np.asarray(trajectory["states"][state_index], dtype=float),
        float(trajectory["timesteps"][step_index]),
        float(trajectory["previous_timesteps"][step_index]),
        c4f13._history(trajectory, step_index),
    )


def _reaction_from_cache(cache) -> CausalFiveFieldFixedQReaction:
    array_names = (
        "q3_value",
        "q3_physical_derivative",
        "q3_scaled_derivative",
        "q3_derivative_norms",
        "descriptor_scaled_matrix",
        "reaction_scaled_rows",
        "reaction_lift",
        "reaction_physical_rows",
        "reaction_physical_ledger",
        "raw_reaction_scaled_rows",
        "raw_reaction_lift",
        "raw_schur_inverse",
        "support_envelope",
    )
    arrays = {name: np.asarray(cache[name]) for name in array_names}
    return CausalFiveFieldFixedQReaction(
        **arrays,
        support_cell_indices=np.asarray(
            cache["support_cell_indices"], dtype=int
        ),
        maximum_descriptor_reconstruction_defect=float(
            cache["maximum_descriptor_reconstruction_defect"]
        ),
        maximum_descriptor_partition_defect=float(
            cache["maximum_descriptor_partition_defect"]
        ),
        maximum_identity_defect=float(cache["maximum_identity_defect"]),
        maximum_reaction_ledger_relative_defect=float(
            cache["maximum_reaction_ledger_relative_defect"]
        ),
        maximum_reaction_support_relative_defect=float(
            cache["maximum_reaction_support_relative_defect"]
        ),
    )


def _data() -> dict:
    (
        layout,
        configuration,
        trajectory,
        step_index,
        old,
        state,
        timestep,
        previous_timestep,
        history,
    ) = _second_state_endpoint()
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(
        state.shape
    )
    rows = np.asarray(configuration["rows"], dtype=float).reshape(state.shape)
    reaction = None
    tangent = None
    if BASE_REFERENCE_PATH.exists():
        with np.load(BASE_REFERENCE_PATH, allow_pickle=False) as cache:
            if np.array_equal(cache["endpoint_primitive_charts"], state):
                reaction = _reaction_from_cache(cache)
                tangent = SimpleNamespace(
                    scaled_base_rate_per_s=np.asarray(
                        cache["scaled_base_rate_per_s"]
                    ),
                    evolving_scaled_jacobian=np.asarray(
                        cache["evolving_scaled_jacobian"]
                    ),
                )
                print("c4f24c: reused 16 ms base cache", flush=True)
    if reaction is None or tangent is None:
        reaction = causal_five_field_fixed_q_reaction(
            context,
            state,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            parent_cell_indices=layout.parent_cell_indices,
            refinement_ratio=layout.refinement_ratio,
        )
        tangent = causal_five_field_monolithic_frozen_tangent(
            context,
            state,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
        )
        payload = {
            name: getattr(reaction, name)
            for name in reaction.__dataclass_fields__
        }
        payload.update(
            endpoint_primitive_charts=state,
            scaled_base_rate_per_s=tangent.scaled_base_rate_per_s,
            evolving_scaled_jacobian=tangent.evolving_scaled_jacobian,
        )
        _write_npz(BASE_REFERENCE_PATH, **payload)
        print("c4f24c: wrote 16 ms base cache", flush=True)
    multiplier = -reaction.q3_scaled_derivative @ tangent.scaled_base_rate_per_s
    rate = tangent.scaled_base_rate_per_s + reaction.reaction_lift @ multiplier
    return {
        "layout": layout,
        "trajectory": trajectory,
        "step_index": step_index,
        "old": old,
        "state": state,
        "timestep": timestep,
        "previous_timestep": previous_timestep,
        "history": history,
        "context": context,
        "columns": columns,
        "rows": rows,
        "reaction": reaction,
        "tangent": tangent,
        "multiplier": multiplier,
        "rate": rate,
    }


def _derivative_audit() -> dict:
    data = _data()
    direction = c4f24b._direction()
    layout = data["layout"]
    state = data["state"]
    columns = data["columns"]
    rows = data["rows"]
    reaction = data["reaction"]
    multiplier = data["multiplier"]
    dimensions = int(state.size)
    matrix = causal_five_field_fixed_q_augmented_step_matrix(
        data["context"],
        data["old"],
        state,
        multiplier,
        data["timestep"],
        data["previous_timestep"],
        order=2,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=layout.parent_cell_indices,
        refinement_ratio=layout.refinement_ratio,
        constraint_row_scales=reaction.q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=reaction.raw_schur_inverse,
        reaction=reaction,
    )
    step = 1.0e-4
    augmented_values = []
    monolithic_values = []
    raw_reaction_values = []
    began = time.perf_counter()
    for coefficient in (1.0, -1.0, 2.0, -2.0):
        candidate = state + coefficient * step * columns * direction[
            :dimensions
        ].reshape(state.shape)
        evaluation = evaluate_causal_five_field_fixed_q_bdf(
            data["old"],
            candidate,
            multiplier + coefficient * step * direction[dimensions:],
            reaction.q3_value,
            data["timestep"],
            data["context"],
            order=2,
            history=data["history"],
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            parent_cell_indices=layout.parent_cell_indices,
            refinement_ratio=layout.refinement_ratio,
            constraint_row_scales=reaction.q3_derivative_norms,
            reaction_channel_basis="frozen_normalized",
            reaction_channel_transform=reaction.raw_schur_inverse,
        )
        augmented_values.append(evaluation.augmented_scaled_residual)
        monolithic_values.append(evaluation.scaled_monolithic_residual)
        raw_reaction_values.append(
            evaluation.reaction.raw_reaction_scaled_rows
        )
        print(
            f"c4f24c: complete residual sample {coefficient:+.0f}h",
            flush=True,
        )
    stencil = lambda values: (  # noqa: E731
        -values[2] + 8.0 * values[0] - 8.0 * values[1] + values[3]
    ) / (12.0 * step)
    direct_augmented = stencil(augmented_values)
    direct_monolithic = stencil(monolithic_values)
    direct_raw_reaction = stencil(raw_reaction_values)
    analytic_augmented = matrix.scaled_matrix @ direction
    analytic_monolithic = (
        matrix.monolithic_scaled_matrix @ direction[:dimensions]
    )
    raw_multiplier = reaction.raw_schur_inverse @ multiplier
    direct_raw_action = direct_raw_reaction @ raw_multiplier
    analytic_raw_action = (
        matrix.reaction_state_scaled_matrix @ direction[:dimensions]
    )
    metrics = {
        "state_time_seconds": SECOND_STATE_TIME_SECONDS,
        "relative_step": step,
        "direct_monolithic_JVP_relative_defect": _relative(
            analytic_monolithic, direct_monolithic
        ),
        "direct_augmented_JVP_relative_defect": _relative(
            analytic_augmented, direct_augmented
        ),
        "direct_raw_reaction_JVP_relative_defect": _relative(
            analytic_raw_action, direct_raw_action
        ),
        "maximum_augmented_block_closure_defect": (
            matrix.maximum_block_closure_defect
        ),
        "maximum_reaction_ledger_relative_defect": (
            matrix.maximum_reaction_ledger_relative_defect
        ),
        "wall_seconds": time.perf_counter() - began,
    }
    metrics["passed"] = bool(
        metrics["direct_monolithic_JVP_relative_defect"]
        <= GATES["maximum_direct_monolithic_JVP_relative_defect"]
        and metrics["direct_augmented_JVP_relative_defect"]
        <= GATES["maximum_direct_augmented_JVP_relative_defect"]
        and metrics["direct_raw_reaction_JVP_relative_defect"]
        <= GATES["maximum_direct_raw_reaction_JVP_relative_defect"]
    )
    _write_npz(
        DERIVATIVE_PATH,
        direction=direction,
        analytic_augmented_JVP=analytic_augmented,
        direct_augmented_JVP=direct_augmented,
        analytic_monolithic_JVP=analytic_monolithic,
        direct_monolithic_JVP=direct_monolithic,
        analytic_raw_reaction_action=analytic_raw_action,
        direct_raw_reaction_action=direct_raw_action,
    )
    _write_json(CHECKPOINT_DIRECTORY / "second_state_derivative.json", metrics)
    return metrics


def _equal_q_history(data: dict, timestep: float):
    state = data["state"]
    columns = data["columns"]
    reaction = data["reaction"]
    scaled_increment = -timestep * data["rate"]
    target = reaction.q3_value
    target_scale = np.maximum(np.abs(target), np.finfo(float).tiny)
    exterior_face = 36 * int(data["layout"].refinement_ratio)
    used = 0
    for iteration in range(12):
        previous = state + columns * scaled_increment.reshape(state.shape)
        local_reaction = causal_five_field_fixed_q_reaction(
            data["context"],
            previous,
            primitive_column_scales=columns,
            conservation_row_scales=data["rows"],
            parent_cell_indices=data["layout"].parent_cell_indices,
            refinement_ratio=data["layout"].refinement_ratio,
        )
        q3 = local_reaction.q3_value
        defect = float(np.max(np.abs((q3 - target) / target_scale)))
        used = iteration + 1
        print(
            "c4f24c: equal-Q history "
            f"dt={timestep:.3e} iteration={used} defect={defect:.6e}",
            flush=True,
        )
        if defect <= 1.0e-13:
            break
        correction = local_reaction.reaction_lift @ (
            (q3 - target) / local_reaction.q3_derivative_norms
        )
        accepted = False
        for line_search in range(8):
            fraction = 0.5**line_search
            candidate_increment = scaled_increment - fraction * correction
            candidate = state + columns * candidate_increment.reshape(
                state.shape
            )
            candidate_q3, _candidate_factors = (
                causal_five_field_exterior_q3(
                    data["context"],
                    candidate,
                    exterior_face_index=exterior_face,
                )
            )
            candidate_defect = float(
                np.max(np.abs((candidate_q3 - target) / target_scale))
            )
            if candidate_defect < defect:
                scaled_increment = candidate_increment
                accepted = True
                print(
                    "c4f24c: equal-Q history line search "
                    f"fraction={fraction:.6e} "
                    f"defect={candidate_defect:.6e}",
                    flush=True,
                )
                break
        if not accepted:
            raise RuntimeError("equal-Q history projection failed to contract")
    previous = state + columns * scaled_increment.reshape(state.shape)
    q3, _factors = causal_five_field_exterior_q3(
        data["context"], previous, exterior_face_index=exterior_face
    )
    defect = float(np.max(np.abs((q3 - target) / target_scale)))
    storage = causal_five_field_monolithic_storage_increment(
        data["context"], previous, state
    )
    history = CausalFiveFieldMonolithicBDFHistory(
        previous_primitive_increment=state - previous,
        previous_mapped_storage_increment=storage.mapped_path_increment,
        previous_responsive_height_storage_increment=(
            storage.responsive_height_path_increment
        ),
        previous_timestep_seconds=timestep,
    ).validated(n_cells=state.shape[0])
    return previous, history, defect, used


def _step_path(index: int) -> Path:
    return CHECKPOINT_DIRECTORY / f"bdf2_step_{int(index)}.npz"


def _step(index: int) -> dict:
    selected = int(index)
    if not 0 <= selected < STEP_TIMESTEPS_SECONDS.size:
        raise ValueError("BDF2 step index is invalid")
    data = _data()
    timestep = float(STEP_TIMESTEPS_SECONDS[selected])
    previous, history, history_defect, history_iterations = _equal_q_history(
        data, timestep
    )
    coefficient = 1.5
    top_left = (
        coefficient * data["reaction"].descriptor_scaled_matrix / timestep
        + data["tangent"].evolving_scaled_jacobian
    )
    began = time.perf_counter()
    result = solve_causal_five_field_fixed_q_bdf(
        data["context"],
        data["state"],
        timestep,
        data["rate"],
        data["multiplier"],
        top_left,
        order=2,
        history=history,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        q3_target=data["reaction"].q3_value,
        constraint_row_scales=data["reaction"].q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=data["reaction"].raw_schur_inverse,
        residual_tolerance=GATES["maximum_exact_step_scaled_residual"],
        constraint_tolerance=GATES["maximum_exact_step_Q3_relative_defect"],
        maximum_newton_iterations=5,
        maximum_line_search_iterations=4,
        refresh_exact_jacobian=True,
        maximum_exact_jacobian_refreshes=(4 if selected < 3 else 1),
        base_reaction=data["reaction"],
        progress_callback=lambda payload: print(
            f"c4f24c step {selected}: {payload}", flush=True
        ),
    )
    metrics = {
        "step_index": selected,
        "timestep_seconds": timestep,
        "accepted": result.accepted,
        "message": result.message,
        "iterations": result.iterations,
        "function_evaluations": result.function_evaluations,
        "maximum_scaled_residual": result.maximum_scaled_residual,
        "maximum_Q3_relative_defect": (
            result.evaluation.maximum_constraint_relative_defect
        ),
        "history_Q3_relative_defect": history_defect,
        "history_projection_iterations": history_iterations,
        "maximum_exact_Jacobian_refreshes": (4 if selected < 3 else 1),
        "rate_relative_defect": _relative(
            result.scaled_rate_per_s, data["rate"]
        ),
        "multiplier_relative_defect": _relative(
            result.multipliers, data["multiplier"]
        ),
        "direct_storage_rate_used": (
            result.evaluation.monolithic_evaluation
            .temporal_storage_uses_direct_rate_action
        ),
        "wall_seconds": time.perf_counter() - began,
    }
    _write_npz(
        _step_path(selected),
        previous_primitive_charts=previous,
        previous_primitive_increment=history.previous_primitive_increment,
        previous_mapped_storage_increment=(
            history.previous_mapped_storage_increment
        ),
        previous_responsive_height_storage_increment=(
            history.previous_responsive_height_storage_increment
        ),
        primitive_charts=result.primitive_charts,
        scaled_bdf_rate_per_s=result.scaled_rate_per_s,
        scaled_interval_rate_per_s=result.scaled_interval_rate_per_s,
        multipliers=result.multipliers,
        augmented_scaled_residual=result.evaluation.augmented_scaled_residual,
        metrics_json=np.asarray(json.dumps(metrics)),
    )
    _write_json(CHECKPOINT_DIRECTORY / f"bdf2_step_{selected}.json", metrics)
    return metrics


def _order(coarse: float, fine: float) -> float:
    return float(math.log(coarse / fine) / math.log(2.0))


def _finalize() -> dict:
    derivative = json.loads(
        (CHECKPOINT_DIRECTORY / "second_state_derivative.json").read_text()
    )
    steps = [
        json.loads(
            (CHECKPOINT_DIRECTORY / f"bdf2_step_{index}.json").read_text()
        )
        for index in range(STEP_TIMESTEPS_SECONDS.size)
    ]
    for step in steps:
        step.setdefault(
            "maximum_exact_Jacobian_refreshes",
            4 if int(step["step_index"]) < 3 else 1,
        )
    small_timestep_rate_orders = [
        _order(
            steps[index]["rate_relative_defect"],
            steps[index + 1]["rate_relative_defect"],
        )
        for index in range(2)
    ]
    small_timestep_multiplier_orders = [
        _order(
            steps[index]["multiplier_relative_defect"],
            steps[index + 1]["multiplier_relative_defect"],
        )
        for index in range(2)
    ]
    rate_orders = [
        _order(
            steps[coarse]["rate_relative_defect"],
            steps[fine]["rate_relative_defect"],
        )
        for coarse, fine in zip(
            BINDING_STEP_INDICES[:-1],
            BINDING_STEP_INDICES[1:],
            strict=True,
        )
    ]
    multiplier_orders = [
        _order(
            steps[coarse]["multiplier_relative_defect"],
            steps[fine]["multiplier_relative_defect"],
        )
        for coarse, fine in zip(
            BINDING_STEP_INDICES[:-1],
            BINDING_STEP_INDICES[1:],
            strict=True,
        )
    ]
    passed = bool(
        derivative["passed"]
        and all(step["accepted"] for step in steps)
        and max(step["history_Q3_relative_defect"] for step in steps)
        <= GATES["maximum_history_Q3_relative_defect"]
        and min(rate_orders) >= GATES["minimum_rate_convergence_order"]
        and min(multiplier_orders)
        >= GATES["minimum_multiplier_convergence_order"]
    )
    roots_passed = bool(
        all(step["accepted"] for step in steps)
        and max(step["history_Q3_relative_defect"] for step in steps)
        <= GATES["maximum_history_Q3_relative_defect"]
    )
    limit_orders_passed = bool(
        min(rate_orders) >= GATES["minimum_rate_convergence_order"]
        and min(multiplier_orders)
        >= GATES["minimum_multiplier_convergence_order"]
    )
    summary = {
        "work_package": "WP10c9d6c7c3b5c4f24c",
        "analysis_only": True,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "second_state_time_seconds": SECOND_STATE_TIME_SECONDS,
        "derivative_audit": derivative,
        "exact_bdf2_steps": steps,
        "diagnostic_small_timestep_rate_convergence_orders": (
            small_timestep_rate_orders
        ),
        "diagnostic_small_timestep_multiplier_convergence_orders": (
            small_timestep_multiplier_orders
        ),
        "binding_step_indices": BINDING_STEP_INDICES,
        "rate_convergence_orders": rate_orders,
        "multiplier_convergence_orders": multiplier_orders,
        "gates": GATES,
        "second_state_derivative_certified": derivative["passed"],
        "exact_constrained_BDF2_roots_certified": roots_passed,
        "synthetic_history_limit_orders_certified": limit_orders_passed,
        "passed": passed,
        "classification": (
            "fixed_Q_second_state_and_constrained_BDF2_preflight_passed"
            if passed
            else (
                "fixed_Q_second_state_Jacobian_and_exact_BDF2_roots_passed_"
                "but_synthetic_history_limit_orders_failed"
            )
        ),
        "authorized_next": (
            "definitions_only_one_Q_execution_manifest"
            if passed
            else "definitions_only_constrained_BDF_startup_history_preflight"
        ),
        "one_Q_execution_manifest_authorized": passed,
        "fixed_Q_micro_solver_authorized": False,
        "one_Q_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    _write_json(CHECKPOINT_DIRECTORY / "summary.json", summary)
    return summary


def _publish() -> dict:
    summary = _finalize()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CANONICAL_DIRECTORY / "summary.json",
        summary,
    )
    _write_json(
        CANONICAL_DIRECTORY / "config.json",
        {
            "work_package": summary["work_package"],
            "second_state_time_seconds": SECOND_STATE_TIME_SECONDS,
            "bdf2_timesteps_seconds": STEP_TIMESTEPS_SECONDS,
            "binding_step_indices": BINDING_STEP_INDICES,
            "reaction_channel_basis": "frozen_normalized",
            "gates": GATES,
        },
    )
    with np.load(DERIVATIVE_PATH, allow_pickle=False) as derivative:
        arrays = {
            f"derivative_{name}": np.asarray(derivative[name])
            for name in derivative.files
        }
    arrays["timesteps_seconds"] = STEP_TIMESTEPS_SECONDS
    for index in range(STEP_TIMESTEPS_SECONDS.size):
        with np.load(_step_path(index), allow_pickle=False) as step:
            for name in step.files:
                if name != "metrics_json":
                    arrays[f"step_{index}_{name}"] = np.asarray(step[name])
    arrays["rate_convergence_orders"] = np.asarray(
        summary["rate_convergence_orders"]
    )
    arrays["multiplier_convergence_orders"] = np.asarray(
        summary["multiplier_convergence_orders"]
    )
    _write_npz(CANONICAL_DIRECTORY / "decisive_arrays.npz", **arrays)
    source_files = (
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
        "scripts/run_causal_inner_face36_fixed_q_second_state_bdf2_preflight_"
        "wp10c9d6c7c3b5c4f24c.py",
        "tests/test_causal_inner_fixed_q.py",
        "tests/test_causal_inner_monolithic_bdf.py",
        "tests/test_causal_inner_face36_fixed_q_second_state_bdf2_preflight_"
        "wp10c9d6c7c3b5c4f24c.py",
        REPORT_RELATIVE,
    )
    source_hashes = {name: _sha(ROOT / name) for name in source_files}
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": 1,
            "execution_base_commit": _git("rev-parse", "HEAD"),
            "implementation_commit": None,
            "working_tree_clean": False,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "source_hashes": source_hashes,
            "implementation_source_bundle_sha256": hashlib.sha256(
                "\n".join(
                    f"{name}  {digest}"
                    for name, digest in sorted(source_hashes.items())
                ).encode()
            ).hexdigest(),
        },
    )
    checksum_names = (
        "config.json",
        "decisive_arrays.npz",
        "provenance.json",
        "summary.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in checksum_names
        )
    )
    catalog = json.loads(CANONICAL_SUMMARY.read_text())
    catalog["artifacts"][ARTIFACT] = {
        "classification": summary["classification"],
        "passed": summary["passed"],
        "path": f"results/canonical/{ARTIFACT}",
    }
    _write_json(CANONICAL_SUMMARY, catalog)
    with CANONICAL_MANIFEST.open(newline="") as handle:
        records = list(csv.DictReader(handle))
        fieldnames = list(records[0])
    records = [record for record in records if record["case"] != ARTIFACT]
    baseline = list(
        csv.DictReader(
            io.StringIO(
                _git(
                    "show",
                    f"HEAD:{CANONICAL_MANIFEST.relative_to(ROOT)}",
                )
            )
        )
    )
    by_key = {(record["case"], record["path"]): record for record in records}
    ordered = []
    for record in baseline:
        key = (record["case"], record["path"])
        if key in by_key:
            ordered.append(by_key.pop(key))
    ordered.extend(by_key.values())
    records = ordered
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            records.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": (
                        "SUPPORTED BUT NOT FULLY CERTIFIED"
                        if summary["passed"]
                        else "REJECTED"
                    ),
                }
            )
    temporary = CANONICAL_MANIFEST.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(CANONICAL_MANIFEST)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derivative-audit", action="store_true")
    parser.add_argument("--step-index", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--publish", action="store_true")
    arguments = parser.parse_args()
    selected = sum(
        (
            arguments.derivative_audit,
            arguments.step_index is not None,
            arguments.finalize,
            arguments.publish,
        )
    )
    if selected != 1:
        parser.error("select exactly one execution stage")
    if arguments.derivative_audit:
        result = _derivative_audit()
    elif arguments.step_index is not None:
        result = _step(arguments.step_index)
    elif arguments.finalize:
        result = _finalize()
    else:
        result = _publish()
    print(json.dumps(c4f24b._plain(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
