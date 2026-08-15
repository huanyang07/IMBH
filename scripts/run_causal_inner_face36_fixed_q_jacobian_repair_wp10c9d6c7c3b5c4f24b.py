#!/usr/bin/env python3
"""Run the repaired fixed-Q Jacobian and exact-step consistency audit.

The committed middle 20 ms endpoint is reused.  This runner advances no
trajectory: it differentiates one committed BDF2 endpoint and solves a small
ladder of independent constrained backward-Euler steps from that endpoint.
Each expensive rung is checkpointed separately.
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

import run_causal_inner_face36_leading_two_plus_hmm_fixed_q_preflight_wp10c9d6c7c3b5c4f22 as c4f22  # noqa: E402
import run_causal_inner_face36_state_dependent_fixed_q_step_preflight_wp10c9d6c7c3b5c4f24 as c4f24  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    CausalFiveFieldFixedQReaction,
    causal_five_field_fixed_q_augmented_step_matrix,
    causal_five_field_fixed_q_reaction,
    evaluate_causal_five_field_fixed_q_bdf,
    solve_causal_five_field_fixed_q_backward_euler,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24b"
ARTIFACT = (
    "causal_inner_face36_fixed_q_jacobian_repair_"
    "wp10c9d6c7c3b5c4f24b"
)
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
BASE_REFERENCE_PATH = CHECKPOINT_DIRECTORY / "base_reference.npz"
DERIVATIVE_PATH = CHECKPOINT_DIRECTORY / "derivative_audit.npz"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_FACE36_FIXED_Q_JACOBIAN_REPAIR_"
    "WP10C9D6C7C3B5C4F24B_2026-08-14.md"
)
STEP_TIMESTEPS_SECONDS = np.asarray((1.0e-7, 5.0e-8, 2.5e-8), dtype=float)
GATES = {
    "maximum_direct_monolithic_JVP_relative_defect": 1.0e-8,
    "maximum_direct_augmented_JVP_relative_defect": 1.0e-8,
    "maximum_direct_raw_reaction_JVP_relative_defect": 1.0e-8,
    "maximum_exact_step_scaled_residual": 1.0e-10,
    "maximum_exact_step_Q3_relative_defect": 1.0e-12,
    "minimum_rate_convergence_order": 0.9,
    "minimum_multiplier_convergence_order": 0.9,
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _data():
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
    ) = c4f24._endpoint_data()
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(
        new.shape
    )
    rows = np.asarray(configuration["rows"], dtype=float).reshape(new.shape)
    reaction = None
    tangent = None
    if BASE_REFERENCE_PATH.exists():
        with np.load(BASE_REFERENCE_PATH, allow_pickle=False) as cache:
            if np.array_equal(cache["endpoint_primitive_charts"], new):
                reaction = CausalFiveFieldFixedQReaction(
                    q3_value=np.asarray(cache["q3_value"], dtype=float),
                    q3_physical_derivative=np.asarray(
                        cache["q3_physical_derivative"], dtype=float
                    ),
                    q3_scaled_derivative=np.asarray(
                        cache["q3_scaled_derivative"], dtype=float
                    ),
                    q3_derivative_norms=np.asarray(
                        cache["q3_derivative_norms"], dtype=float
                    ),
                    descriptor_scaled_matrix=np.asarray(
                        cache["descriptor_scaled_matrix"], dtype=float
                    ),
                    reaction_scaled_rows=np.asarray(
                        cache["reaction_scaled_rows"], dtype=float
                    ),
                    reaction_lift=np.asarray(
                        cache["reaction_lift"], dtype=float
                    ),
                    reaction_physical_rows=np.asarray(
                        cache["reaction_physical_rows"], dtype=float
                    ),
                    reaction_physical_ledger=np.asarray(
                        cache["reaction_physical_ledger"], dtype=float
                    ),
                    raw_reaction_scaled_rows=np.asarray(
                        cache["raw_reaction_scaled_rows"], dtype=float
                    ),
                    raw_reaction_lift=np.asarray(
                        cache["raw_reaction_lift"], dtype=float
                    ),
                    raw_schur_inverse=np.asarray(
                        cache["raw_schur_inverse"], dtype=float
                    ),
                    support_cell_indices=np.asarray(
                        cache["support_cell_indices"], dtype=int
                    ),
                    support_envelope=np.asarray(
                        cache["support_envelope"], dtype=float
                    ),
                    maximum_descriptor_reconstruction_defect=float(
                        cache["maximum_descriptor_reconstruction_defect"]
                    ),
                    maximum_descriptor_partition_defect=float(
                        cache["maximum_descriptor_partition_defect"]
                    ),
                    maximum_identity_defect=float(
                        cache["maximum_identity_defect"]
                    ),
                    maximum_reaction_ledger_relative_defect=float(
                        cache["maximum_reaction_ledger_relative_defect"]
                    ),
                    maximum_reaction_support_relative_defect=float(
                        cache["maximum_reaction_support_relative_defect"]
                    ),
                )
                tangent = SimpleNamespace(
                    scaled_base_rate_per_s=np.asarray(
                        cache["scaled_base_rate_per_s"], dtype=float
                    ),
                    evolving_scaled_jacobian=np.asarray(
                        cache["evolving_scaled_jacobian"], dtype=float
                    ),
                )
                print("c4f24b: reused base reference cache", flush=True)
    if reaction is None or tangent is None:
        reaction = causal_five_field_fixed_q_reaction(
            context,
            new,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            parent_cell_indices=layout.parent_cell_indices,
            refinement_ratio=layout.refinement_ratio,
        )
        tangent = causal_five_field_monolithic_frozen_tangent(
            context,
            new,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
        )
        _write_npz(
            BASE_REFERENCE_PATH,
            endpoint_primitive_charts=new,
            q3_value=reaction.q3_value,
            q3_physical_derivative=reaction.q3_physical_derivative,
            q3_scaled_derivative=reaction.q3_scaled_derivative,
            q3_derivative_norms=reaction.q3_derivative_norms,
            descriptor_scaled_matrix=reaction.descriptor_scaled_matrix,
            reaction_scaled_rows=reaction.reaction_scaled_rows,
            reaction_lift=reaction.reaction_lift,
            reaction_physical_rows=reaction.reaction_physical_rows,
            reaction_physical_ledger=reaction.reaction_physical_ledger,
            raw_reaction_scaled_rows=reaction.raw_reaction_scaled_rows,
            raw_reaction_lift=reaction.raw_reaction_lift,
            raw_schur_inverse=reaction.raw_schur_inverse,
            support_cell_indices=reaction.support_cell_indices,
            support_envelope=reaction.support_envelope,
            maximum_descriptor_reconstruction_defect=np.asarray(
                reaction.maximum_descriptor_reconstruction_defect
            ),
            maximum_descriptor_partition_defect=np.asarray(
                reaction.maximum_descriptor_partition_defect
            ),
            maximum_identity_defect=np.asarray(
                reaction.maximum_identity_defect
            ),
            maximum_reaction_ledger_relative_defect=np.asarray(
                reaction.maximum_reaction_ledger_relative_defect
            ),
            maximum_reaction_support_relative_defect=np.asarray(
                reaction.maximum_reaction_support_relative_defect
            ),
            scaled_base_rate_per_s=tangent.scaled_base_rate_per_s,
            evolving_scaled_jacobian=tangent.evolving_scaled_jacobian,
        )
        print("c4f24b: wrote base reference cache", flush=True)
    multiplier = (
        -reaction.q3_scaled_derivative @ tangent.scaled_base_rate_per_s
    )
    rate = (
        tangent.scaled_base_rate_per_s
        + reaction.reaction_lift @ multiplier
    )
    return {
        "layout": layout,
        "trajectory": trajectory,
        "index": index,
        "old": old,
        "new": new,
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


def _direction() -> np.ndarray:
    with np.load(c4f22.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        lifts = np.asarray(arrays["middle_equal_Q_lifts"], dtype=float)
    if lifts.ndim != 2 or lifts.shape[1] < 1:
        raise RuntimeError("c4f24b equal-Q lift matrix is invalid")
    state = np.array(lifts[:, 0], copy=True)
    state /= np.linalg.norm(state)
    multiplier = np.asarray((0.25, -0.125, 0.0625), dtype=float)
    multiplier /= np.linalg.norm(multiplier)
    return np.concatenate((state, multiplier))


def _derivative_audit() -> dict:
    direction = _direction()
    data = _data()
    layout = data["layout"]
    new = data["new"]
    columns = data["columns"]
    rows = data["rows"]
    reaction = data["reaction"]
    multiplier = data["multiplier"]
    dimensions = int(new.size)
    matrix = causal_five_field_fixed_q_augmented_step_matrix(
        data["context"],
        data["old"],
        new,
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
        candidate = new + coefficient * step * columns * direction[
            :dimensions
        ].reshape(new.shape)
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
            "c4f24b: direct residual sample "
            f"{coefficient:+.0f}h complete",
            flush=True,
        )
    direct_augmented = (
        -augmented_values[2]
        + 8.0 * augmented_values[0]
        - 8.0 * augmented_values[1]
        + augmented_values[3]
    ) / (12.0 * step)
    direct_monolithic = (
        -monolithic_values[2]
        + 8.0 * monolithic_values[0]
        - 8.0 * monolithic_values[1]
        + monolithic_values[3]
    ) / (12.0 * step)
    direct_raw_reaction = (
        -raw_reaction_values[2]
        + 8.0 * raw_reaction_values[0]
        - 8.0 * raw_reaction_values[1]
        + raw_reaction_values[3]
    ) / (12.0 * step)
    analytic_augmented = matrix.scaled_matrix @ direction
    analytic_monolithic = (
        matrix.monolithic_scaled_matrix @ direction[:dimensions]
    )
    # The stored reaction-state block is already contracted with the
    # multiplier.  Compare the independent raw-channel derivative through
    # its action reconstructed from the direct samples instead.
    raw_action_direct = direct_raw_reaction @ (
        reaction.raw_schur_inverse @ multiplier
    )
    raw_action_analytic = (
        matrix.reaction_state_scaled_matrix @ direction[:dimensions]
    )
    metrics = {
        "relative_step": step,
        "direction_index": 0,
        "direct_monolithic_JVP_relative_defect": _relative(
            analytic_monolithic,
            direct_monolithic,
        ),
        "direct_augmented_JVP_relative_defect": _relative(
            analytic_augmented,
            direct_augmented,
        ),
        "direct_raw_reaction_JVP_relative_defect": _relative(
            raw_action_analytic,
            raw_action_direct,
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
        analytic_raw_reaction_action=raw_action_analytic,
        direct_raw_reaction_action=raw_action_direct,
    )
    _write_json(CHECKPOINT_DIRECTORY / "derivative_audit.json", metrics)
    return metrics


def _step_path(index: int) -> Path:
    return CHECKPOINT_DIRECTORY / f"exact_step_{int(index)}.npz"


def _step_progress_path(index: int) -> Path:
    return CHECKPOINT_DIRECTORY / f"exact_step_{int(index)}_progress.npz"


def _exact_step(index: int) -> dict:
    selected = int(index)
    if not 0 <= selected < STEP_TIMESTEPS_SECONDS.size:
        raise ValueError("exact-step index is invalid")
    data = _data()
    layout = data["layout"]
    reaction = data["reaction"]
    timestep = float(STEP_TIMESTEPS_SECONDS[selected])
    top_left = (
        reaction.descriptor_scaled_matrix / timestep
        + data["tangent"].evolving_scaled_jacobian
    )
    progress_path = _step_progress_path(selected)
    initial_increment = None
    initial_multiplier = data["multiplier"]
    if progress_path.exists():
        with np.load(progress_path, allow_pickle=False) as checkpoint:
            initial_increment = np.asarray(
                checkpoint["scaled_increment"],
                dtype=float,
            )
            initial_multiplier = np.asarray(
                checkpoint["multipliers"],
                dtype=float,
            )
        print(
            f"c4f24b step {selected}: resumed iteration checkpoint",
            flush=True,
        )

    def progress(payload: dict) -> None:
        print(f"c4f24b step {selected}: {payload}", flush=True)

    def checkpoint(
        iteration: int,
        scaled_increment: np.ndarray,
        multipliers: np.ndarray,
        evaluation,
    ) -> None:
        _write_npz(
            progress_path,
            iteration=np.asarray(iteration),
            scaled_increment=scaled_increment,
            multipliers=multipliers,
            augmented_scaled_residual=evaluation.augmented_scaled_residual,
        )

    began = time.perf_counter()
    result = solve_causal_five_field_fixed_q_backward_euler(
        data["context"],
        data["new"],
        timestep,
        data["rate"],
        initial_multiplier,
        top_left,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=layout.parent_cell_indices,
        refinement_ratio=layout.refinement_ratio,
        q3_target=reaction.q3_value,
        constraint_row_scales=reaction.q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=reaction.raw_schur_inverse,
        residual_tolerance=GATES["maximum_exact_step_scaled_residual"],
        constraint_tolerance=GATES["maximum_exact_step_Q3_relative_defect"],
        maximum_newton_iterations=4,
        maximum_line_search_iterations=1,
        refresh_exact_jacobian=True,
        maximum_exact_jacobian_refreshes=3,
        progress_callback=progress,
        initial_scaled_increment=initial_increment,
        checkpoint_callback=checkpoint,
        base_reaction=reaction,
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
        "rate_relative_defect": _relative(
            result.scaled_rate_per_s,
            data["rate"],
        ),
        "multiplier_relative_defect": _relative(
            result.multipliers,
            data["multiplier"],
        ),
        "wall_seconds": time.perf_counter() - began,
    }
    _write_npz(
        _step_path(selected),
        primitive_charts=result.primitive_charts,
        scaled_rate_per_s=result.scaled_rate_per_s,
        multipliers=result.multipliers,
        continuous_scaled_rate_per_s=data["rate"],
        continuous_multipliers=data["multiplier"],
        augmented_scaled_residual=(
            result.evaluation.augmented_scaled_residual
        ),
        metrics_json=np.asarray(json.dumps(_plain(metrics))),
    )
    _write_json(
        CHECKPOINT_DIRECTORY / f"exact_step_{selected}.json",
        metrics,
    )
    return metrics


def _order(coarse: float, fine: float) -> float:
    return float(np.log(coarse / fine) / np.log(2.0))


def _finalize() -> dict:
    derivative = json.loads(
        (CHECKPOINT_DIRECTORY / "derivative_audit.json").read_text(
            encoding="utf-8"
        )
    )
    steps = []
    for index in range(STEP_TIMESTEPS_SECONDS.size):
        path = CHECKPOINT_DIRECTORY / f"exact_step_{index}.json"
        if not path.exists():
            raise RuntimeError(f"missing exact-step checkpoint {index}")
        steps.append(json.loads(path.read_text(encoding="utf-8")))
    rate_orders = [
        _order(steps[index]["rate_relative_defect"], steps[index + 1]["rate_relative_defect"])
        for index in range(len(steps) - 1)
    ]
    multiplier_orders = [
        _order(
            steps[index]["multiplier_relative_defect"],
            steps[index + 1]["multiplier_relative_defect"],
        )
        for index in range(len(steps) - 1)
    ]
    passed = bool(
        derivative["passed"]
        and all(step["accepted"] for step in steps)
        and min(rate_orders) >= GATES["minimum_rate_convergence_order"]
        and min(multiplier_orders)
        >= GATES["minimum_multiplier_convergence_order"]
    )
    summary = {
        "work_package": WORK_PACKAGE,
        "analysis_only": True,
        "trajectory_executed": False,
        "physical_operator_changed": False,
        "reaction_channel_basis": "frozen_normalized",
        "derivative_audit": derivative,
        "exact_steps": steps,
        "rate_convergence_orders": rate_orders,
        "multiplier_convergence_orders": multiplier_orders,
        "gates": GATES,
        "passed": passed,
        "classification": (
            "fixed_Q_Jacobian_and_exact_BE_limit_repair_passed"
            if passed
            else "fixed_Q_Jacobian_or_exact_BE_limit_repair_failed"
        ),
        "one_Q_execution_manifest_authorized": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "second_state_and_constrained_BDF2_preflight"
            if passed
            else None
        ),
    }
    _write_json(CHECKPOINT_DIRECTORY / "summary.json", summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True), flush=True)
    return summary


def _publish() -> dict:
    summary_path = CHECKPOINT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        raise RuntimeError("finalized c4f24b summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("passed"):
        raise RuntimeError("failed c4f24b result cannot be published as passing")
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    canonical_summary = CANONICAL_DIRECTORY / "summary.json"
    canonical_config = CANONICAL_DIRECTORY / "config.json"
    canonical_arrays = CANONICAL_DIRECTORY / "decisive_arrays.npz"
    canonical_provenance = CANONICAL_DIRECTORY / "provenance.json"
    _write_json(canonical_summary, summary)
    _write_json(
        canonical_config,
        {
            "schema_version": 1,
            "work_package": WORK_PACKAGE,
            "reaction_channel_basis": "frozen_normalized",
            "direct_JVP_relative_step": 1.0e-4,
            "exact_BE_timesteps_seconds": STEP_TIMESTEPS_SECONDS,
            "gates": GATES,
            "analysis_only": True,
            "trajectory_executed": False,
            "physical_operator_changed": False,
        },
    )
    with np.load(DERIVATIVE_PATH, allow_pickle=False) as derivative:
        arrays = {
            f"derivative_{key}": np.asarray(derivative[key])
            for key in derivative.files
        }
    rate_defects = []
    multiplier_defects = []
    for index in range(STEP_TIMESTEPS_SECONDS.size):
        with np.load(_step_path(index), allow_pickle=False) as step:
            for key in (
                "primitive_charts",
                "scaled_rate_per_s",
                "multipliers",
                "augmented_scaled_residual",
            ):
                arrays[f"step_{index}_{key}"] = np.asarray(step[key])
        metrics = summary["exact_steps"][index]
        rate_defects.append(metrics["rate_relative_defect"])
        multiplier_defects.append(metrics["multiplier_relative_defect"])
    arrays.update(
        {
            "timesteps_seconds": STEP_TIMESTEPS_SECONDS,
            "rate_relative_defects": np.asarray(rate_defects),
            "multiplier_relative_defects": np.asarray(multiplier_defects),
            "rate_convergence_orders": np.asarray(
                summary["rate_convergence_orders"]
            ),
            "multiplier_convergence_orders": np.asarray(
                summary["multiplier_convergence_orders"]
            ),
        }
    )
    _write_npz(canonical_arrays, **arrays)
    source_files = (
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_dae.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_discrete_tangent.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_linear_tangent.py",
        "scripts/run_causal_inner_face36_fixed_q_jacobian_repair_wp10c9d6c7c3b5c4f24b.py",
        "tests/test_causal_inner_fixed_q.py",
        "tests/test_causal_inner_monolithic_discrete_tangent.py",
        "tests/test_causal_inner_face36_fixed_q_jacobian_repair_"
        "wp10c9d6c7c3b5c4f24b.py",
        REPORT_RELATIVE,
    )
    source_hashes = {relative: _sha(ROOT / relative) for relative in source_files}
    source_bundle = hashlib.sha256(
        "\n".join(
            f"{relative}  {digest}"
            for relative, digest in sorted(source_hashes.items())
        ).encode("utf-8")
    ).hexdigest()
    _write_json(
        canonical_provenance,
        {
            "schema_version": 1,
            "execution_base_commit": _git("rev-parse", "HEAD"),
            "execution_base_tree": _git("rev-parse", "HEAD^{tree}"),
            "working_tree_clean": False,
            "implementation_commit": None,
            "implementation_source_bundle_sha256": source_bundle,
            "source_hashes": source_hashes,
            "predecessor_summary_sha256": _sha(c4f24.SUMMARY_PATH),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "commands": [
                f"python {Path(__file__).name} --derivative-audit",
                *[
                    f"python {Path(__file__).name} --step-index {index}"
                    for index in range(STEP_TIMESTEPS_SECONDS.size)
                ],
                f"python {Path(__file__).name} --finalize",
            ],
            "focused_test_command": (
                "python -m pytest -q tests/test_causal_inner_fixed_q.py "
                "tests/test_causal_inner_monolithic_discrete_tangent.py "
                "tests/test_causal_inner_face36_state_dependent_fixed_q_"
                "step_preflight_wp10c9d6c7c3b5c4f24.py"
            ),
            "focused_test_result": "15 passed in 88.25 s",
        },
    )
    checksum_path = CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in (
                "config.json",
                "decisive_arrays.npz",
                "provenance.json",
                "summary.json",
            )
        ),
        encoding="utf-8",
    )
    catalog_summary = json.loads(CANONICAL_SUMMARY.read_text(encoding="utf-8"))
    catalog_summary["artifacts"][ARTIFACT] = {
        "classification": summary["classification"],
        "passed": True,
        "path": f"results/canonical/{ARTIFACT}",
    }
    _write_json(CANONICAL_SUMMARY, catalog_summary)
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
        fieldnames = list(records[0])
    records = [record for record in records if record["case"] != ARTIFACT]
    records_by_key = {
        (record["case"], record["path"]): record for record in records
    }
    baseline_text = _git(
        "show",
        f"HEAD:{CANONICAL_MANIFEST.relative_to(ROOT)}",
    )
    baseline_records = list(csv.DictReader(io.StringIO(baseline_text)))
    ordered_records = []
    for baseline_record in baseline_records:
        key = (baseline_record["case"], baseline_record["path"])
        if key in records_by_key:
            ordered_records.append(records_by_key.pop(key))
    ordered_records.extend(records_by_key.values())
    records = ordered_records
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            records.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
                }
            )
    temporary_manifest = CANONICAL_MANIFEST.with_suffix(".tmp.csv")
    with temporary_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    temporary_manifest.replace(CANONICAL_MANIFEST)
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
        raise SystemExit("select exactly one audit action")
    if arguments.derivative_audit:
        print(json.dumps(_plain(_derivative_audit()), indent=2), flush=True)
    elif arguments.step_index is not None:
        print(
            json.dumps(_plain(_exact_step(arguments.step_index)), indent=2),
            flush=True,
        )
    elif arguments.finalize:
        _finalize()
    else:
        print(json.dumps(_plain(_publish()), indent=2), flush=True)


if __name__ == "__main__":
    main()
