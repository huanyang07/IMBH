#!/usr/bin/env python3
"""Execute the frozen iteration-reserve primary continuation retry."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

e14d = importlib.import_module(
    "run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d"
)
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (  # noqa: E402
    causal_five_field_fixed_q_continuation_state,
    causal_five_field_fixed_q_continuation_states_equal,
    load_causal_five_field_fixed_q_continuation_state,
    solve_causal_five_field_fixed_q_bdf,
)


WORK_PACKAGE = "WP10c9d6c7c3b5c4f24e14l"
ARTIFACT = (
    "causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l"
)
MANIFEST_ARTIFACT = (
    "causal_inner_face36_fixed_q_primary_retry_manifest_"
    "wp10c9d6c7c3b5c4f24e14k"
)
MANIFEST_DIRECTORY = ROOT / "results/canonical" / MANIFEST_ARTIFACT
SEED_PATH = e14d.SEED_PATH
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_fixed_q_primary_retry_"
    "wp10c9d6c7c3b5c4f24e14l.py"
)
SOURCE_FILES = (
    THIS_RUNNER,
    THIS_TEST,
    "scripts/run_causal_inner_face36_fixed_q_primary_bounded_continuation_"
    "wp10c9d6c7c3b5c4f24e14d.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_fixed_q.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_bdf.py",
    "scripts/run_causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1.py",
    "scripts/run_causal_inner_face36_fixed_q_primary_retry_manifest_"
    "wp10c9d6c7c3b5c4f24e14k.py",
)
THREAD_ENVIRONMENT = e14d.THREAD_ENVIRONMENT
TIMESTEP_SECONDS = e14d.TIMESTEP_SECONDS
HALF_TIMESTEP_SECONDS = e14d.HALF_TIMESTEP_SECONDS


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _root_policy(label: str) -> dict:
    cold = label in {"cold_1", "cold_shadow", "half_1", "half_2"}
    return {
        "cold": cold,
        "initial_exact_jacobian_required": cold,
        "maximum_exact_jacobian_refreshes": 2 if cold else 1,
        "use_carried_solver_state": not cold,
        "exact_jacobian_refresh_policy": (
            "on_line_search_failure"
            if cold
            else "on_line_search_failure_or_iteration_reserve"
        ),
    }


@contextmanager
def _legacy_runtime():
    replacements = {
        "WORK_PACKAGE": WORK_PACKAGE,
        "ARTIFACT": ARTIFACT,
        "MANIFEST_ARTIFACT": MANIFEST_ARTIFACT,
        "MANIFEST_DIRECTORY": MANIFEST_DIRECTORY,
        "SEED_PATH": SEED_PATH,
        "SCRATCH_DIRECTORY": SCRATCH_DIRECTORY,
        "CANONICAL_DIRECTORY": CANONICAL_DIRECTORY,
        "THIS_RUNNER": THIS_RUNNER,
        "THIS_TEST": THIS_TEST,
        "SOURCE_FILES": SOURCE_FILES,
        "_root_policy": _root_policy,
        "_solve_root": _solve_root,
    }
    original = {name: getattr(e14d, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(e14d, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(e14d, name, value)


def _validate_frozen_contract(*, require_clean: bool) -> dict:
    with _legacy_runtime():
        frozen = e14d._validate_frozen_contract(require_clean=require_clean)
    contract = frozen["contract"]
    parent = _read(MANIFEST_DIRECTORY / "parent_lock.json")
    if (
        contract["solver_contract"]["warm_refresh_policy"]
        != "on_line_search_failure_or_iteration_reserve"
        or contract["solver_contract"]["warm_iteration_reserve_trigger"] != 6
        or contract["solver_contract"][
            "warm_failed_relative_backtrack_trigger"
        ]
        != 4
        or contract["solver_contract"][
            "maximum_exact_assemblies_per_warm_root"
        ]
        != 1
        or contract["trajectory_gates"][
            "same_history_warm_to_cold_wall_ratio_maximum"
        ]
        != 0.75
        or parent["summary"]["classification"]
        != "warm_policy_scientific_and_cost_passed"
        or not parent["summary"]["full_primary_retry_manifest_authorized"]
    ):
        raise RuntimeError("iteration-reserve primary retry contract changed")
    return frozen


def _solve_root(
    label: str,
    data: dict,
    continuation,
    timestep: float,
    identity: dict,
    *,
    artifact_label: str | None = None,
):
    stored_label = label if artifact_label is None else artifact_label
    policy = _root_policy(label)
    rate, multiplier = e14d._predictors(continuation, data["columns"])
    top_left = (
        e14d._cold_top_left(data, continuation, timestep)
        if policy["cold"]
        else None
    )
    carried = (
        continuation.nonlinear_solver_state
        if policy["use_carried_solver_state"]
        else None
    )
    events = []

    def progress(payload: dict) -> None:
        plain = e14d._plain(payload)
        events.append(plain)
        print(f"e14l {label}: {plain}", flush=True)

    began_wall = time.perf_counter()
    began_process = time.process_time()
    result = solve_causal_five_field_fixed_q_bdf(
        data["context"],
        continuation.current_primitive_charts,
        timestep,
        rate,
        multiplier,
        top_left,
        order=2,
        history=continuation.history,
        primitive_column_scales=data["columns"],
        conservation_row_scales=data["rows"],
        parent_cell_indices=data["layout"].parent_cell_indices,
        refinement_ratio=data["layout"].refinement_ratio,
        q3_target=continuation.q3_target,
        constraint_row_scales=continuation.constraint_row_scales,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=continuation.next_reaction_channel_transform,
        residual_tolerance=1.0e-10,
        constraint_tolerance=1.0e-12,
        ledger_tolerance=1.0e-12,
        storage_parity_tolerance=1.0e-9,
        minimum_reconstruction_factor=1.0 - 1.0e-12,
        maximum_schur_condition_number=1.0e8,
        maximum_scaled_primitive_change=5.0e-3,
        maximum_newton_iterations=8,
        maximum_line_search_iterations=12,
        refresh_exact_jacobian=True,
        maximum_exact_jacobian_refreshes=(
            policy["maximum_exact_jacobian_refreshes"]
        ),
        exact_jacobian_refresh_policy=policy["exact_jacobian_refresh_policy"],
        initial_nonlinear_solver_state=carried,
        initial_exact_jacobian_required=(
            policy["initial_exact_jacobian_required"]
        ),
        solver_state_provenance=identity,
        physical_state_audit=e14d.e1._state_audit,
        require_physical_state_audit=True,
        maximum_h_over_r=0.12,
        minimum_scattering_optical_depth=1.0,
        progress_callback=progress,
    )
    wall = time.perf_counter() - began_wall
    process = time.process_time() - began_process
    metrics = e14d._result_metrics(result, events, wall, process)
    metrics.update(
        {"label": label, "policy": policy, "timestep_seconds": timestep}
    )
    e14d._save_result(
        SCRATCH_DIRECTORY / f"result_{stored_label}.npz", result, metrics
    )
    e14d._write_json(
        SCRATCH_DIRECTORY / f"metrics_{stored_label}.json", metrics
    )
    return result, metrics


def _canonicalize(metrics: dict, data: dict, main_results: dict) -> None:
    e14d._canonicalize(metrics, data, main_results)
    summary_path = CANONICAL_DIRECTORY / "summary.json"
    summary = _read(summary_path)
    summary.update(
        {
            "parent_classification_preserved": "bounded_continuation_failed",
            "iteration_reserve_warm_policy_used": True,
            "full_primary_retry_completed": bool(metrics["scientific_passed"]),
        }
    )
    e14d._write_json(summary_path, summary)
    files = [
        path
        for path in sorted(CANONICAL_DIRECTORY.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{e14d._sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    e14d._catalog(summary)


def _run_impl() -> dict:
    frozen = _validate_frozen_contract(require_clean=True)
    identity = e14d._execution_identity()
    if SCRATCH_DIRECTORY.exists():
        raise RuntimeError("primary retry scratch directory already exists")
    SCRATCH_DIRECTORY.mkdir(parents=True)
    e14d._write_json(SCRATCH_DIRECTORY / "execution_identity.json", identity)
    data = e14d.e1._state_data("primary_20ms")
    continuation = e14d._load_seed(data, identity)
    main_results = {}
    main_continuations = {}
    main_metrics = {}
    start_checkpoints = {"cold_1": continuation}
    scientific_passed = True
    failure_stage = None
    try:
        for label in ("cold_1", "warm_1", "warm_2", "warm_3"):
            start_checkpoints[label] = continuation
            result, continuation, metrics = e14d._advance(
                label, data, continuation, TIMESTEP_SECONDS, identity
            )
            main_results[label] = result
            main_continuations[label] = continuation
            main_metrics[label] = metrics
    except e14d.BindingRootFailure as error:
        main_results[error.label] = error.result
        main_metrics[error.label] = error.metrics
        scientific_passed = False
        failure_stage = str(error)

    replay_metrics = {"executed": False, "passed": False}
    shadow_metrics = {"executed": False, "scientific_passed": False}
    half_metrics = {"executed": False, "passed": False}
    if scientific_passed:
        replay_start = load_causal_five_field_fixed_q_continuation_state(
            SCRATCH_DIRECTORY / "checkpoint_warm_1.npz",
            data["context"],
            expected_provenance=identity,
        )
        replay_bitwise = True
        replay_roots = {}
        replay_continuation = replay_start
        for label in ("warm_2", "warm_3"):
            result, metrics = _solve_root(
                label,
                data,
                replay_continuation,
                TIMESTEP_SECONDS,
                identity,
                artifact_label=f"replay_{label}",
            )
            if (
                not result.accepted
                or result.exact_jacobian_assemblies
                > _root_policy(label)["maximum_exact_jacobian_refreshes"]
            ):
                replay_bitwise = False
                replay_roots[label] = {
                    "result_bitwise": False,
                    "continuation_bitwise": False,
                    "accepted": result.accepted,
                    "metrics": metrics,
                }
                break
            next_replay = causal_five_field_fixed_q_continuation_state(
                result,
                data["context"],
                replay_continuation.current_primitive_charts,
                primitive_column_scales=data["columns"],
                conservation_row_scales=data["rows"],
                parent_cell_indices=data["layout"].parent_cell_indices,
                refinement_ratio=data["layout"].refinement_ratio,
                elapsed_time_seconds=(
                    replay_continuation.elapsed_time_seconds + TIMESTEP_SECONDS
                ),
                completed_steps=replay_continuation.completed_steps + 1,
                provenance=identity,
            )
            root_equal = e14d._bitwise_results_equal(
                main_results[label], result, main_metrics[label], metrics
            )
            continuation_equal = causal_five_field_fixed_q_continuation_states_equal(
                main_continuations[label], next_replay
            )
            replay_bitwise = bool(
                replay_bitwise and root_equal and continuation_equal
            )
            replay_roots[label] = {
                "result_bitwise": root_equal,
                "continuation_bitwise": continuation_equal,
                "accepted": result.accepted,
                "metrics": metrics,
            }
            replay_continuation = next_replay
        replay_metrics = {
            "executed": True,
            "passed": replay_bitwise,
            "roots": replay_roots,
        }
        scientific_passed = bool(scientific_passed and replay_bitwise)
        if not replay_bitwise:
            failure_stage = "bitwise_suffix_replay"

    if scientific_passed:
        shadow_start = start_checkpoints["warm_2"]
        shadow, shadow_root = _solve_root(
            "cold_shadow", data, shadow_start, TIMESTEP_SECONDS, identity
        )
        state_defect = e14d._scaled_state_absolute(
            shadow.primitive_charts,
            main_results["warm_2"].primitive_charts,
            data["columns"],
        )
        action_defect = e14d._relative(
            shadow.scaled_reaction_rate_action_per_s,
            main_results["warm_2"].scaled_reaction_rate_action_per_s,
        )
        wall_ratio = main_metrics["warm_2"]["root_wall_seconds"] / max(
            shadow_root["root_wall_seconds"], np.finfo(float).tiny
        )
        residual_ratio = main_metrics["warm_2"]["function_evaluations"] / max(
            shadow_root["function_evaluations"], 1
        )
        shadow_scientific = bool(
            shadow.accepted
            and shadow.exact_jacobian_assemblies <= 2
            and state_defect <= 1.0e-8
            and action_defect <= 1.0e-8
        )
        shadow_metrics = {
            "executed": True,
            "scientific_passed": shadow_scientific,
            "scaled_state_absolute_defect": state_defect,
            "reaction_action_relative_defect": action_defect,
            "warm_to_cold_wall_time_ratio": wall_ratio,
            "warm_to_cold_residual_evaluation_ratio": residual_ratio,
            "cost_passed": wall_ratio <= 0.75,
            "root": shadow_root,
        }
        scientific_passed = bool(scientific_passed and shadow_scientific)
        if not shadow_scientific:
            failure_stage = "same_history_cold_shadow"

    if replay_metrics["passed"]:
        half_start = start_checkpoints["warm_3"]
        try:
            _, half_continuation, half_1_metrics = e14d._advance(
                "half_1", data, half_start, HALF_TIMESTEP_SECONDS, identity
            )
            half_2, _, half_2_metrics = e14d._advance(
                "half_2",
                data,
                half_continuation,
                HALF_TIMESTEP_SECONDS,
                identity,
            )
            state_defect = e14d._scaled_endpoint_difference(
                half_2.primitive_charts,
                main_results["warm_3"].primitive_charts,
                half_start.current_primitive_charts,
                data["columns"],
            )
            action_defect = e14d._relative(
                half_2.scaled_reaction_rate_action_per_s,
                main_results["warm_3"].scaled_reaction_rate_action_per_s,
            )
            half_passed = bool(state_defect <= 0.1 and action_defect <= 0.1)
            half_metrics = {
                "executed": True,
                "passed": half_passed,
                "state_difference_relative_to_full_step_change": state_defect,
                "reaction_action_relative_difference": action_defect,
                "half_1": half_1_metrics,
                "half_2": half_2_metrics,
            }
        except e14d.BindingRootFailure as error:
            half_passed = False
            half_metrics = {
                "executed": True,
                "passed": False,
                "failed_root": error.label,
                "failed_root_metrics": error.metrics,
            }
        scientific_passed = bool(scientific_passed and half_passed)
        if not half_passed:
            failure_stage = "matched_endpoint_half_step_audit"

    accounting = e14d._failure_aware_root_accounting(main_metrics)
    cumulative_ledger = accounting["accepted_trajectory_cumulative_ledger"]
    ledger_passed = bool(
        accounting["planned_ladder_complete"] and cumulative_ledger <= 4.0e-12
    )
    scientific_passed = bool(scientific_passed and ledger_passed)
    if not ledger_passed and failure_stage is None:
        failure_stage = "cumulative_ledger"
    cost_passed = bool(
        scientific_passed and shadow_metrics.get("cost_passed", False)
    )
    classification = e14d._classification(scientific_passed, cost_passed)
    warm_labels = [
        label for label in ("warm_1", "warm_2", "warm_3") if label in main_metrics
    ]
    total_main_wall = sum(
        root["root_wall_seconds"] for root in main_metrics.values()
    )
    accepted_horizon = accounting["accepted_trajectory_horizon_seconds"]
    metrics = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "scientific_passed": scientific_passed,
        "cost_passed": cost_passed,
        "failure_stage": failure_stage,
        "main_roots": main_metrics,
        "replay": replay_metrics,
        "same_history_cold_shadow": shadow_metrics,
        "matched_endpoint_half_step_audit": half_metrics,
        "root_accounting": accounting,
        "cumulative_absolute_ledger_defect": cumulative_ledger,
        "cumulative_ledger_complete": accounting["planned_ladder_complete"],
        "cumulative_ledger_passed": ledger_passed,
        "warm_exact_assembly_count": sum(
            main_metrics[label]["exact_Jacobian_assemblies"]
            for label in warm_labels
        ),
        "warm_residual_evaluations": sum(
            main_metrics[label]["function_evaluations"] for label in warm_labels
        ),
        "main_root_wall_seconds": total_main_wall,
        "accepted_physical_seconds_per_wall_hour": (
            accepted_horizon * 3600.0
            / max(total_main_wall, np.finfo(float).tiny)
        ),
        "identity": identity,
        "frozen_manifest": frozen["summary"],
        "parent_classification_preserved": "bounded_continuation_failed",
    }
    e14d._write_json(SCRATCH_DIRECTORY / "execution_metrics.json", metrics)
    _canonicalize(metrics, data, main_results)
    return metrics


def _run() -> dict:
    with _legacy_runtime():
        return _run_impl()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.validate:
        print(
            json.dumps(
                e14d._plain(_validate_frozen_contract(require_clean=False)),
                indent=2,
                sort_keys=True,
            )
        )
        return
    metrics = _run()
    print(json.dumps(e14d._plain(metrics), indent=2, sort_keys=True))
    if not metrics["scientific_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
