#!/usr/bin/env python3
"""Recover the full-bundle transient with prospectively adaptive AB2 steps."""

from __future__ import annotations

import argparse
import csv
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_adaptive_full_bundle_transient_recovery_manifest_wp10c9d6c7c3b5c4f25fizt as parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_atlas import (  # noqa: E402
    conservative_ledger_relative_defect,
    pack_macro_outputs,
    truth_outputs_from_radial_operator,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_integrator import (  # noqa: E402
    macro_rate_output_matrix,
    reconstruct_thermodynamic_macro_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_radial import (  # noqa: E402
    generalized_maxwell_cattaneo_radial_operator,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_thermodynamic_macro_atlas import (  # noqa: E402
    thermodynamic_macro_chart_pullback,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = (
    "WP10c9d6c7c3b5c4f25fizu_"
    "entropy_complete_adaptive_full_bundle_transient_recovery_execution"
)
SLAVING_CLASSIFICATION = (
    "entropy_complete_adaptive_full_bundle_transient_reached_212ms_with_slaving"
)
OPEN_CLASSIFICATION = (
    "entropy_complete_adaptive_full_bundle_transient_reached_212ms_without_slaving"
)
BUDGET_CLASSIFICATION = "entropy_complete_adaptive_transient_recovery_budget_exhausted"
FAIL_CLASSIFICATION = "entropy_complete_adaptive_transient_recovery_failed"
SLAVING_AUTHORIZED_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizv_"
    "entropy_complete_terminal_fast_graph_tangent_certificate_manifest"
)
OPEN_AUTHORIZED_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizv_"
    "entropy_complete_transient_geometry_and_cost_decision_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_adaptive_full_bundle_transient_recovery_"
    "execution_wp10c9d6c7c3b5c4f25fizu"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_ADAPTIVE_FULL_"
    "BUNDLE_TRANSIENT_RECOVERY_EXECUTION_"
    "WP10C9D6C7C3B5C4F25FIZU_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_adaptive_full_bundle_transient_"
    "recovery_execution_wp10c9d6c7c3b5c4f25fizu.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_adaptive_full_bundle_transient_"
    "recovery_execution_wp10c9d6c7c3b5c4f25fizu.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "dee8e0d20b986c8786cc473f64ee7bd3ae046100704028c5a8950dcf3f1c3d96"
)
PRECANONICALIZATION_INVOCATION_COMMIT = (
    "01364fb734ead80f22b3d9937abaeaee8ab342b9"
)
PARENT_ARRAYS = parent.PARENT_ARRAYS
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _relative(defect, *references) -> float:
    return parent.parent._relative(defect, *references)


def _variable_ab2_candidate(
    current_state: np.ndarray,
    current_rate: np.ndarray,
    previous_rate: np.ndarray,
    timestep_seconds: float,
    previous_timestep_seconds: float,
) -> np.ndarray:
    state = np.asarray(current_state, dtype=float)
    current = np.asarray(current_rate, dtype=float)
    previous = np.asarray(previous_rate, dtype=float)
    timestep = float(timestep_seconds)
    old_timestep = float(previous_timestep_seconds)
    if (
        any(item.shape != (16, 5) for item in (state, current, previous))
        or any(np.any(~np.isfinite(item)) for item in (state, current, previous))
        or not np.isfinite(timestep)
        or not np.isfinite(old_timestep)
        or timestep <= 0.0
        or old_timestep <= 0.0
    ):
        raise ValueError("variable-step AB2 inputs are invalid")
    ratio = timestep / old_timestep
    return state + timestep * (
        (1.0 + 0.5 * ratio) * current - 0.5 * ratio * previous
    )


def _variable_ab2_integral(
    current_output: np.ndarray,
    previous_output: np.ndarray,
    timestep_seconds: float,
    previous_timestep_seconds: float,
) -> np.ndarray:
    current = np.asarray(current_output, dtype=float)
    previous = np.asarray(previous_output, dtype=float)
    timestep = float(timestep_seconds)
    old_timestep = float(previous_timestep_seconds)
    if (
        current.shape != (115,)
        or previous.shape != current.shape
        or np.any(~np.isfinite(current))
        or np.any(~np.isfinite(previous))
        or timestep <= 0.0
        or old_timestep <= 0.0
    ):
        raise ValueError("variable-step AB2 output inputs are invalid")
    ratio = timestep / old_timestep
    return timestep * (
        (1.0 + 0.5 * ratio) * current - 0.5 * ratio * previous
    )


def _bitwise_roundtrip(arrays: dict[str, np.ndarray]) -> bool:
    stream = BytesIO()
    np.savez(stream, **arrays)
    stream.seek(0)
    with np.load(stream) as loaded:
        return set(loaded.files) == set(arrays) and all(
            np.array_equal(np.asarray(arrays[name]), np.asarray(loaded[name]))
            for name in arrays
        )


def _is_hyperbolicity_failure(exception: Exception) -> bool:
    return "not real within the declared tolerance" in str(exception)


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if PARENT_CHECKSUM_MANIFEST_SHA256 == "TO_BE_FROZEN":
        raise RuntimeError("adaptive recovery parent checksum is not frozen")
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("adaptive recovery manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "adaptive_recovery_contract.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["fixed_step_rejection_preserved"]
        or not summary["adaptive_recovery_execution_authorized"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or contract["bounded_execution"]["maximum_new_truth_operator_calls"]
        != 128
    ):
        raise RuntimeError("adaptive transient recovery authorization changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"adaptive recovery manifest source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("adaptive recovery execution needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _execute() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    contract = validated["contract"]
    adaptive = contract["adaptive_AB2"]
    bounded = contract["bounded_execution"]
    gates = contract["binding_gates"]
    with np.load(PARENT_ARRAYS) as archive:
        seed = {name: np.asarray(archive[name]) for name in archive.files}
    state_history = [np.array(value, copy=True) for value in seed["accepted_macro_states"]]
    chart_history = [
        np.array(value, copy=True) for value in seed["accepted_primitive_charts"]
    ]
    output_history = [
        np.array(value, copy=True)
        for value in seed["accepted_truth_packed_outputs"]
    ]
    rate_history = [
        np.array(value, copy=True)
        for value in seed["accepted_macro_rates_per_second"]
    ]
    integrated_history = [
        np.array(value, copy=True)
        for value in seed["accepted_integrated_packed_outputs"]
    ]
    initial_checkpoint = {
        "previous_state": state_history[-2],
        "current_state": state_history[-1],
        "previous_charts": chart_history[-2],
        "current_charts": chart_history[-1],
        "previous_output": output_history[-2],
        "current_output": output_history[-1],
        "previous_rate": rate_history[-2],
        "current_rate": rate_history[-1],
        "elapsed_seconds": np.asarray(
            [bounded["initial_absolute_elapsed_seconds"]], dtype=float
        ),
        "previous_timestep_seconds": np.asarray(
            [contract["restart"]["previous_timestep_seconds"]], dtype=float
        ),
    }
    initial_roundtrip = _bitwise_roundtrip(initial_checkpoint)
    truth_execution = parent.parent.parent.parent.rejected_execution.truth_execution
    truth_source = truth_execution.truth_source
    physical_gates = truth_source.fixed_q_implementation.parent._contract()[
        "binding_physical_gates"
    ]
    context_start = time.perf_counter()
    context, _profile, _initial = truth_source.fixed_q_implementation._primary_setup()
    context_seconds = time.perf_counter() - context_start
    rate_output = macro_rate_output_matrix()
    current_time = float(bounded["initial_absolute_elapsed_seconds"])
    previous_timestep = float(contract["restart"]["previous_timestep_seconds"])
    timestep = float(adaptive["initial_timestep_seconds"])
    attempted_steps = 0
    truth_calls = 0
    retry_count = 0
    accepted_new_steps = 0
    low_defect_streak = 0
    slaving_streak = 0
    maximum_slaving_streak = 0
    records = []
    stop_reason = None
    physical_failure = False
    execution_start = time.perf_counter()
    while current_time < bounded["target_absolute_elapsed_seconds"] - 1.0e-15:
        if attempted_steps >= bounded["maximum_attempted_steps"]:
            stop_reason = "attempt_budget_exhausted"
            break
        if truth_calls >= bounded["maximum_new_truth_operator_calls"]:
            stop_reason = "truth_call_budget_exhausted"
            break
        attempted_steps += 1
        timestep = min(
            timestep, bounded["target_absolute_elapsed_seconds"] - current_time
        )
        step_start = time.perf_counter()
        current_state = state_history[-1]
        current_charts = chart_history[-1]
        anchor_macro, coordinate_scales, _tangents, pullbacks = (
            thermodynamic_macro_chart_pullback(
                context, current_charts, derivative_step=1.0e-5
            )
        )
        candidate_state = _variable_ab2_candidate(
            current_state,
            rate_history[-1],
            rate_history[-2],
            timestep,
            previous_timestep,
        )
        record = {
            "attempt": attempted_steps,
            "candidate_timestep_seconds": timestep,
            "candidate_absolute_elapsed_seconds": current_time + timestep,
            "timestep_ratio": timestep / previous_timestep,
            "anchor_roundtrip_relative_defect": _relative(
                anchor_macro - current_state, anchor_macro, current_state
            ),
        }
        try:
            reconstruction = reconstruct_thermodynamic_macro_state(
                context,
                current_charts,
                candidate_state,
                anchor_macro_state=anchor_macro,
                macro_coordinate_scales=coordinate_scales,
                macro_coordinate_pullbacks=pullbacks,
                derivative_step=1.0e-5,
                maximum_newton_corrections=8,
                relative_tolerance=gates[
                    "maximum_macro_roundtrip_relative_defect"
                ],
                maximum_chart_coordinate_infinity=gates[
                    "reserved_reconstruction_chart_coordinate"
                ],
            )
        except Exception as exc:
            record.update(
                {
                    "accepted": False,
                    "retryable": True,
                    "stage": "pretruth_reconstruction",
                    "exception": f"{type(exc).__name__}: {exc}",
                    "truth_call_performed": False,
                    "wall_seconds": time.perf_counter() - step_start,
                }
            )
            records.append(record)
            next_timestep = timestep * adaptive["shrink_factor"]
            print(
                f"adaptive attempt {attempted_steps}: pretruth retry "
                f"h={timestep:.6e} -> {next_timestep:.6e}",
                flush=True,
            )
            if next_timestep < adaptive["minimum_timestep_seconds"] - 1.0e-18:
                stop_reason = "minimum_timestep_reconstruction_failure"
                break
            timestep = next_timestep
            retry_count += 1
            low_defect_streak = 0
            continue
        maximum_chart = float(np.max(np.abs(reconstruction.chart_coordinates)))
        truth_call_started = False
        try:
            prefilter = (
                truth_source.adaptive_diagnosis._midpoint_hyperbolicity_audit(
                    context, reconstruction.primitive_charts
                )
            )
            truth_calls += 1
            truth_call_started = True
            operator = generalized_maxwell_cattaneo_radial_operator(
                context, reconstruction.primitive_charts, quadrature_order=8
            )
            physical = truth_source._operator_record(operator)
            physical_checks = truth_source._physical_checks(
                physical, physical_gates
            )
            physical_checks.update(
                {
                    "prefilter_eigenvalue": prefilter[
                        "maximum_eigenvalue_imaginary_ratio"
                    ]
                    <= 1.0e-10,
                    "prefilter_eigenvector": prefilter[
                        "maximum_eigenvector_imaginary_ratio"
                    ]
                    <= 1.0e-10,
                    "hydrostatic_embedding": parent.parent.parent.parent.rejected_execution._hydrostatic_embedding_defect(
                        context, reconstruction.primitive_charts
                    )
                    <= 1.0e-10,
                }
            )
            outputs = truth_outputs_from_radial_operator(operator)
            packed_output = pack_macro_outputs(outputs)
            candidate_rate = (rate_output @ packed_output).reshape(16, 5)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            hyperbolicity_failure = _is_hyperbolicity_failure(exc)
            record.update(
                {
                    "accepted": False,
                    "retryable": False,
                    "stage": "physical_truth_evaluation",
                    "exception": message,
                    "truth_call_performed": truth_call_started,
                    "truth_call": truth_calls if truth_call_started else None,
                    "maximum_chart_coordinate": maximum_chart,
                    "macro_roundtrip_relative_defect": reconstruction.maximum_macro_state_roundtrip_relative_defect,
                    "reconstruction_newton_corrections": reconstruction.newton_corrections,
                    "physical_hyperbolicity_failure": hyperbolicity_failure,
                    "wall_seconds": time.perf_counter() - step_start,
                }
            )
            records.append(record)
            stop_reason = (
                "physical_hyperbolicity_truth_gate_failed"
                if hyperbolicity_failure
                else "truth_evaluation_failed_closed"
            )
            physical_failure = hyperbolicity_failure
            print(
                f"adaptive attempt {attempted_steps}: nonretryable truth "
                f"failure at t={current_time + timestep:.6e}: {message}",
                flush=True,
            )
            break
        embedded = parent.parent._embedded_defect(
            current_state,
            candidate_state,
            rate_history[-1],
            candidate_rate,
            coordinate_scales,
            timestep,
        )
        integrated_output = _variable_ab2_integral(
            output_history[-1], output_history[-2], timestep, previous_timestep
        )
        actual_change = candidate_state - current_state
        ledger_change = (rate_output @ integrated_output).reshape(16, 5)
        ledger_defect = _relative(
            actual_change - ledger_change, actual_change, ledger_change
        )
        numerical_passed = bool(
            record["anchor_roundtrip_relative_defect"]
            <= gates["maximum_macro_roundtrip_relative_defect"]
            and reconstruction.maximum_macro_state_roundtrip_relative_defect
            <= gates["maximum_macro_roundtrip_relative_defect"]
            and maximum_chart
            <= gates["reserved_reconstruction_chart_coordinate"]
            and embedded
            <= gates["maximum_AB2_trapezoidal_embedded_defect"]
            and ledger_defect
            <= gates["maximum_discrete_conservative_ledger_relative_defect"]
            and conservative_ledger_relative_defect(outputs)
            <= gates["maximum_discrete_conservative_ledger_relative_defect"]
        )
        physical_passed = all(physical_checks.values())
        slaving = parent.parent._slaving_record(
            candidate_rate, coordinate_scales, contract
        )
        record.update(
            {
                "truth_call": truth_calls,
                "truth_call_performed": True,
                "maximum_chart_coordinate": maximum_chart,
                "macro_roundtrip_relative_defect": reconstruction.maximum_macro_state_roundtrip_relative_defect,
                "reconstruction_newton_corrections": reconstruction.newton_corrections,
                "embedded_defect": embedded,
                "discrete_ledger_relative_defect": ledger_defect,
                "truth_output_ledger_relative_defect": conservative_ledger_relative_defect(outputs),
                "numerical_passed": numerical_passed,
                "physical": physical,
                "physical_checks": physical_checks,
                "physical_passed": physical_passed,
                "slaving": slaving,
                "wall_seconds": time.perf_counter() - step_start,
            }
        )
        if not physical_passed:
            record.update(
                {
                    "accepted": False,
                    "retryable": False,
                    "stage": "physical_truth_gate",
                }
            )
            records.append(record)
            stop_reason = "physical_truth_gate_failed"
            physical_failure = True
            print(
                f"adaptive attempt {attempted_steps}: physical failure at "
                f"t={current_time + timestep:.6e}",
                flush=True,
            )
            break
        if not numerical_passed:
            record.update(
                {
                    "accepted": False,
                    "retryable": True,
                    "stage": "posttruth_numerical_gate",
                }
            )
            records.append(record)
            next_timestep = timestep * adaptive["shrink_factor"]
            print(
                f"adaptive attempt {attempted_steps}: numerical retry "
                f"h={timestep:.6e} -> {next_timestep:.6e}",
                flush=True,
            )
            if next_timestep < adaptive["minimum_timestep_seconds"] - 1.0e-18:
                stop_reason = "minimum_timestep_numerical_failure"
                break
            timestep = next_timestep
            retry_count += 1
            low_defect_streak = 0
            continue
        if slaving["instantaneous_slaving_observation_passed"]:
            slaving_streak += 1
        else:
            slaving_streak = 0
        maximum_slaving_streak = max(maximum_slaving_streak, slaving_streak)
        accepted_new_steps += 1
        current_time += timestep
        record.update(
            {
                "accepted": True,
                "retryable": False,
                "stage": "accepted",
                "accepted_step": accepted_new_steps,
                "slaving_streak": slaving_streak,
            }
        )
        records.append(record)
        state_history.append(np.array(candidate_state, copy=True))
        chart_history.append(np.array(reconstruction.primitive_charts, copy=True))
        output_history.append(np.array(packed_output, copy=True))
        rate_history.append(np.array(candidate_rate, copy=True))
        integrated_history.append(np.array(integrated_output, copy=True))
        print(
            f"adaptive accepted {accepted_new_steps}: t={current_time:.6e} "
            f"h={timestep:.6e} chart={maximum_chart:.6e} "
            f"embedded={embedded:.6e} "
            f"aux={slaving['normalized_auxiliary_rate_infinity_per_second']:.6e}",
            flush=True,
        )
        previous_timestep = timestep
        if (
            maximum_chart <= adaptive["growth_chart_coordinate_maximum"]
            and embedded <= adaptive["growth_embedded_defect_maximum"]
        ):
            low_defect_streak += 1
        else:
            low_defect_streak = 0
        if low_defect_streak >= adaptive[
            "growth_requires_consecutive_low_defect_steps"
        ]:
            timestep = min(
                adaptive["maximum_timestep_seconds"],
                timestep * adaptive["growth_factor"],
            )
            low_defect_streak = 0
    target_reached = bool(
        abs(current_time - bounded["target_absolute_elapsed_seconds"]) <= 1.0e-14
    )
    required_slaving_streak = contract["slaving_observation"][
        "required_consecutive_accepted_steps"
    ]
    persistent = bool(target_reached and slaving_streak >= required_slaving_streak)
    budget_exhausted = stop_reason in {
        "attempt_budget_exhausted",
        "truth_call_budget_exhausted",
    }
    terminal_checkpoint = {
        "previous_state": state_history[-2],
        "current_state": state_history[-1],
        "previous_charts": chart_history[-2],
        "current_charts": chart_history[-1],
        "previous_output": output_history[-2],
        "current_output": output_history[-1],
        "previous_rate": rate_history[-2],
        "current_rate": rate_history[-1],
        "elapsed_seconds": np.asarray([current_time], dtype=float),
        "previous_timestep_seconds": np.asarray([previous_timestep], dtype=float),
    }
    terminal_roundtrip = _bitwise_roundtrip(terminal_checkpoint)
    restart_passed = initial_roundtrip and terminal_roundtrip
    _terminal_anchor, terminal_coordinate_scales, _terminal_tangents, _terminal_pullbacks = (
        thermodynamic_macro_chart_pullback(
            context, chart_history[-1], derivative_step=1.0e-5
        )
    )
    terminal_slaving = parent.parent._slaving_record(
        rate_history[-1], terminal_coordinate_scales, contract
    )
    passed = target_reached and restart_passed
    classification = (
        SLAVING_CLASSIFICATION
        if persistent and restart_passed
        else OPEN_CLASSIFICATION
        if target_reached and restart_passed
        else BUDGET_CLASSIFICATION
        if budget_exhausted
        else FAIL_CLASSIFICATION
    )
    accepted_records = [record for record in records if record.get("accepted")]
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "stop_reason": stop_reason,
        "physical_failure": physical_failure,
        "target_reached": target_reached,
        "context_construction_wall_seconds": context_seconds,
        "execution_wall_seconds": time.perf_counter() - execution_start,
        "attempted_steps": attempted_steps,
        "accepted_new_steps": accepted_new_steps,
        "retry_count": retry_count,
        "new_truth_operator_calls": truth_calls,
        "accepted_absolute_horizon_seconds": current_time,
        "initial_checkpoint_roundtrip_bitwise": initial_roundtrip,
        "terminal_checkpoint_roundtrip_bitwise": terminal_roundtrip,
        "attempt_records": records,
        "accepted_timestep_seconds": [
            record["candidate_timestep_seconds"] for record in accepted_records
        ],
        "minimum_accepted_timestep_seconds": min(
            (record["candidate_timestep_seconds"] for record in accepted_records),
            default=0.0,
        ),
        "maximum_accepted_timestep_seconds": max(
            (record["candidate_timestep_seconds"] for record in accepted_records),
            default=0.0,
        ),
        "maximum_accepted_chart_coordinate": max(
            (record["maximum_chart_coordinate"] for record in accepted_records),
            default=0.0,
        ),
        "maximum_accepted_embedded_defect": max(
            (record["embedded_defect"] for record in accepted_records),
            default=0.0,
        ),
        "maximum_accepted_discrete_ledger_relative_defect": max(
            (record["discrete_ledger_relative_defect"] for record in accepted_records),
            default=0.0,
        ),
        "maximum_slaving_streak": maximum_slaving_streak,
        "terminal_slaving_streak": slaving_streak,
        "required_terminal_slaving_streak": required_slaving_streak,
        "persistent_auxiliary_slaving_observed": persistent,
        "terminal_slaving": terminal_slaving,
        "new_global_roots": 0,
        "fixed_Q_reaction_calls": 0,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "precanonicalization_invocation": {
            "implementation_commit": PRECANONICALIZATION_INVOCATION_COMMIT,
            "outcome": "uncaught truth hyperbolicity exception; no canonical artifacts written",
        },
    }
    arrays = {
        "accepted_macro_states": np.asarray(state_history),
        "accepted_primitive_charts": np.asarray(chart_history),
        "accepted_truth_packed_outputs": np.asarray(output_history),
        "accepted_macro_rates_per_second": np.asarray(rate_history),
        "accepted_integrated_packed_outputs": np.asarray(integrated_history),
        "terminal_previous_macro_rate_per_second": rate_history[-2],
        "terminal_current_macro_rate_per_second": rate_history[-1],
        "terminal_previous_truth_packed_output": output_history[-2],
        "terminal_current_truth_packed_output": output_history[-1],
        "terminal_previous_timestep_seconds": np.asarray([previous_timestep]),
        "terminal_elapsed_seconds": np.asarray([current_time]),
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": utils._sha256(path),
                    "scientific_status": status,
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
    catalog = utils._read_json(CANONICAL_SUMMARY)
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
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": utils._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("adaptive recovery result already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "adaptive_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "adaptive_arrays.npz", **arrays)
    authorized_next = (
        SLAVING_AUTHORIZED_NEXT
        if metrics["persistent_auxiliary_slaving_observed"]
        else OPEN_AUTHORIZED_NEXT
        if metrics["target_reached"]
        else None
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "target_reached": metrics["target_reached"],
        "accepted_new_steps": metrics["accepted_new_steps"],
        "accepted_absolute_horizon_seconds": metrics[
            "accepted_absolute_horizon_seconds"
        ],
        "persistent_auxiliary_slaving_observed": metrics[
            "persistent_auxiliary_slaving_observed"
        ],
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "parent_arrays_sha256": utils._sha256(PARENT_ARRAYS),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete adaptive full-bundle transient recovery",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"The adaptive restart accepted `{metrics['accepted_new_steps']}` new steps and reached `{metrics['accepted_absolute_horizon_seconds']:.6e}` s using `{metrics['new_truth_operator_calls']}` truth calls and `{metrics['retry_count']}` rejected retries.",
                "",
                f"Accepted timesteps ranged from `{metrics['minimum_accepted_timestep_seconds']:.6e}` to `{metrics['maximum_accepted_timestep_seconds']:.6e}` s. Maximum chart/embedded/ledger defects were `{metrics['maximum_accepted_chart_coordinate']:.6e}`, `{metrics['maximum_accepted_embedded_defect']:.6e}`, and `{metrics['maximum_accepted_discrete_ledger_relative_defect']:.6e}`.",
                "",
                f"Persistent auxiliary slaving observed: `{metrics['persistent_auxiliary_slaving_observed']}`. Complete-cycle execution remains unauthorized.",
                "",
                f"Stop reason: `{metrics['stop_reason']}`. The first invocation at `{PRECANONICALIZATION_INVOCATION_COMMIT}` exposed an uncaught rejection-packaging exception and wrote no canonical artifacts; this rerun changed no scientific gate.",
                "",
                f"Authorized next: `{authorized_next}`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "implementation_commit": utils._git("rev-parse", "HEAD"),
            "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                path: utils._sha256(ROOT / path) for path in sources
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("choose --run")
    metrics, arrays = _execute()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    summary = _canonicalize(metrics, arrays)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
