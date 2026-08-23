#!/usr/bin/env python3
"""Freeze the bounded wide-span autonomous continuation contract.

The contract consumes the committed 16 ms original-free-field trajectory and
the no-new-witness asymptotic diagnosis.  It does not execute a continuation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_complete_cycle_asymptotic_path_diagnosis_wp10c9d6c7c3b5c4f25fg as diagnosis  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fh"
CLASSIFICATION = (
    "autonomous_curvature_adaptive_endpoint_collocation_continuation_"
    "manifest_frozen"
)
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fi_"
    "curvature_adaptive_arclength_continuation_execution"
)
ARTIFACT = (
    "causal_inner_curvature_adaptive_arclength_continuation_manifest_"
    "wp10c9d6c7c3b5c4f25fh"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CURVATURE_ADAPTIVE_ARCLENGTH_"
    "CONTINUATION_MANIFEST_WP10C9D6C7C3B5C4F25FH_2026-08-23.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_curvature_adaptive_arclength_continuation_"
    "manifest_wp10c9d6c7c3b5c4f25fh.py"
)
THIS_TEST = (
    "tests/test_causal_inner_curvature_adaptive_arclength_continuation_"
    "manifest_wp10c9d6c7c3b5c4f25fh.py"
)

INITIAL_SEGMENT_SECONDS = 1.0e-3
MINIMUM_SEGMENT_SECONDS = 2.5e-4
MAXIMUM_SEGMENT_SECONDS = 4.0e-3
MAXIMUM_ACCEPTED_SEGMENTS = 216
MAXIMUM_EXACT_FREE_FIELD_CALLS = 288
BLIND_MIDPOINT_FREQUENCY = 4
MAXIMUM_ENDPOINT_INTEGRAL_DEFECT = 2.0e-2
MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT = 2.0e-2
GROWTH_FACTOR_MAXIMUM = 2.0
ACCEPTED_SEGMENTS_BEFORE_GROWTH = 4
MAXIMUM_EXECUTION_WALL_HOURS = 30.0
COST_RESERVE_FACTOR = 1.25
MAXIMUM_PREVALIDATED_PREDICTOR_DEFECT = 2.0e-2
MAXIMUM_DIAGNOSTIC_FOUR_MILLISECOND_PREDICTOR_DEFECT = 6.0e-2


def _helper():
    return diagnosis._helper()


def _source_paths() -> tuple[Path, ...]:
    return (
        ROOT / THIS_RUNNER,
        ROOT / THIS_TEST,
        ROOT / diagnosis.THIS_RUNNER,
        ROOT / diagnosis.parent.THIS_RUNNER,
        ROOT / "src/imri_qpe/layer3_minidisk_1d/conservative_free_field_rom.py",
    )


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    diagnosis_hashes = helper._validate_checksums(diagnosis.CANONICAL_DIRECTORY)
    summary = helper._read(diagnosis.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        diagnosis.CANONICAL_DIRECTORY / "diagnosis_metrics.json"
    )
    architecture = helper._read(
        diagnosis.CANONICAL_DIRECTORY
        / "mathematical_architecture_decision.json"
    )
    if (
        not summary["passed"]
        or summary["classification"] != diagnosis.CLASSIFICATION
        or not summary["wide_arclength_continuation_manifest_authorized"]
        or summary["authorized_next"] != diagnosis.AUTHORIZED_NEXT
        or not metrics["source_audit"]["truth_field_autonomous"]
        or not metrics["wide_arclength_transport_supported"]
        or metrics["cycle_closure_supported"]
        or metrics["equilibrium_closure_supported"]
        or metrics["gate_values"]["maximum_validated_hermite_stride"] != 16
        or architecture["selected_architecture"]
        != "autonomous_curvature_adaptive_arclength_hermite_continuation"
        or architecture["truth_system"]["external_forcing_phase_added"]
    ):
        raise RuntimeError("asymptotic architecture authorization changed")
    execution_hashes = helper._validate_checksums(
        diagnosis.parent.CANONICAL_DIRECTORY
    )
    execution = helper._read(
        diagnosis.parent.CANONICAL_DIRECTORY / "cycle_execution_metrics.json"
    )
    gates = execution["gate_values"]
    if (
        execution["classification"] != diagnosis.parent.BUDGET_CLASSIFICATION
        or gates["completed_patches"] != 64
        or gates["exact_free_field_witnesses"] != 192
        or gates["cycle_observed"]
    ):
        raise RuntimeError("complete-cycle acquisition evidence changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("wide continuation manifest requires a clean tracked tree")
    return {
        "diagnosis_hashes": diagnosis_hashes,
        "complete_cycle_execution_hashes": execution_hashes,
        "diagnosis_classification": summary["classification"],
        "execution_classification": execution["classification"],
    }


def _cost_projection() -> dict:
    helper = _helper()
    execution = helper._read(
        diagnosis.parent.CANONICAL_DIRECTORY / "cycle_execution_metrics.json"
    )
    values = execution["gate_values"]
    observed_seconds_per_call = (
        float(values["execution_wall_seconds"])
        / int(values["exact_free_field_witnesses"])
    )
    raw_hours = (
        MAXIMUM_EXACT_FREE_FIELD_CALLS * observed_seconds_per_call / 3600.0
    )
    reserved_hours = COST_RESERVE_FACTOR * raw_hours
    no_rejection_calls = (
        1
        + MAXIMUM_ACCEPTED_SEGMENTS
        + MAXIMUM_ACCEPTED_SEGMENTS // BLIND_MIDPOINT_FREQUENCY
    )
    return {
        "observed_parent_wall_seconds_per_exact_call": observed_seconds_per_call,
        "maximum_exact_free_field_calls": MAXIMUM_EXACT_FREE_FIELD_CALLS,
        "maximum_accepted_segments": MAXIMUM_ACCEPTED_SEGMENTS,
        "no_rejection_exact_call_count": no_rejection_calls,
        "rejection_call_reserve": (
            MAXIMUM_EXACT_FREE_FIELD_CALLS - no_rejection_calls
        ),
        "raw_projected_wall_hours": raw_hours,
        "reserve_factor": COST_RESERVE_FACTOR,
        "reserved_projected_wall_hours": reserved_hours,
        "maximum_execution_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
        "cost_gate_passed": reserved_hours <= MAXIMUM_EXECUTION_WALL_HOURS,
        "maximum_horizon_at_one_millisecond_seconds": (
            MAXIMUM_ACCEPTED_SEGMENTS * 1.0e-3
        ),
        "maximum_horizon_at_two_milliseconds_seconds": (
            MAXIMUM_ACCEPTED_SEGMENTS * 2.0e-3
        ),
        "maximum_horizon_at_four_milliseconds_seconds": (
            MAXIMUM_ACCEPTED_SEGMENTS * 4.0e-3
        ),
    }


def _predictor_backtest() -> dict:
    helper = _helper()
    witnesses = helper._load_npz(
        diagnosis.parent.CANONICAL_DIRECTORY / "exact_witness_arrays.npz"
    )
    anchor_mask = np.asarray(witnesses["kinds"]).astype(str) == "anchor"
    coordinates = np.asarray(witnesses["coordinates"], dtype=float)[anchor_mask]
    rates = np.asarray(
        witnesses["coordinate_rates_per_s"], dtype=float
    )[anchor_mask]
    base_step = diagnosis.MACRO_STEP_SECONDS
    cases = {
        "initial_1ms_from_0p25ms_history": (4, 1),
        "steady_1ms": (4, 4),
        "growth_2ms_from_1ms_history": (8, 4),
        "steady_2ms": (8, 8),
        "growth_4ms_from_2ms_history": (16, 8),
        "steady_4ms": (16, 16),
    }
    records = {}
    for name, (span_stride, history_stride) in cases.items():
        defects = []
        span = span_stride * base_step
        history = history_stride * base_step
        for index in range(history_stride, len(coordinates) - span_stride):
            prediction = (
                coordinates[index]
                + span * rates[index]
                + span**2
                / (2.0 * history)
                * (rates[index] - rates[index - history_stride])
            )
            reference = coordinates[index + span_stride]
            displacement = max(
                float(np.linalg.norm(reference - coordinates[index])),
                np.finfo(float).tiny,
            )
            defects.append(
                float(np.linalg.norm(prediction - reference) / displacement)
            )
        values = np.asarray(defects)
        records[name] = {
            "span_seconds": span,
            "history_seconds": history,
            "sample_count": int(len(values)),
            "maximum_relative_endpoint_defect": float(np.max(values)),
            "p95_relative_endpoint_defect": float(np.quantile(values, 0.95)),
        }
    prevalidated = (
        "initial_1ms_from_0p25ms_history",
        "steady_1ms",
        "growth_2ms_from_1ms_history",
        "steady_2ms",
    )
    two_ms_passed = all(
        records[name]["maximum_relative_endpoint_defect"]
        <= MAXIMUM_PREVALIDATED_PREDICTOR_DEFECT
        for name in prevalidated
    )
    four_ms_diagnostic_passed = all(
        records[name]["maximum_relative_endpoint_defect"]
        <= MAXIMUM_DIAGNOSTIC_FOUR_MILLISECOND_PREDICTOR_DEFECT
        for name in ("growth_4ms_from_2ms_history", "steady_4ms")
    )
    return {
        "records": records,
        "maximum_prevalidated_predictor_defect": (
            MAXIMUM_PREVALIDATED_PREDICTOR_DEFECT
        ),
        "two_millisecond_endpoint_proposal_prevalidated": two_ms_passed,
        "four_millisecond_predictor_is_diagnostic_only": True,
        "maximum_diagnostic_four_millisecond_predictor_defect": (
            MAXIMUM_DIAGNOSTIC_FOUR_MILLISECOND_PREDICTOR_DEFECT
        ),
        "four_millisecond_diagnostic_bound_passed": (
            four_ms_diagnostic_passed
        ),
        "passed": bool(two_ms_passed and four_ms_diagnostic_passed),
    }


def _variable_step_ab2_formula() -> dict:
    return {
        "definition": (
            "y_predict=y_n+h*f_n+h^2/(2*h_previous)*(f_n-f_previous)"
        ),
        "derivation": (
            "integrate the linear interpolation of the two most recent exact "
            "endpoint free rates across the proposed physical-time span"
        ),
        "all_rates": "exact original unconstrained reaction-free field",
        "initial_current_rate": (
            "one new exact evaluation at the committed 16 ms terminal state"
        ),
        "initial_previous_rate": (
            "hash-validated exact anchor rate at 15.75 ms"
        ),
        "fixed_Q_rate_used": False,
        "external_phase_used": False,
    }


def _execution_contract(cost: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "definitions_only": True,
        "authorized_execution": AUTHORIZED_NEXT,
        "truth_system": {
            "equation": "dy/dt=f_free(y)",
            "autonomous": True,
            "field": "original unconstrained reaction-free monolithic field",
            "external_clock_or_phase": "forbidden",
            "fixed_Q_rate_or_reaction": "forbidden",
        },
        "path_parameterization": {
            "speed": "nu(y)=||f_free(y)||_2",
            "geometry": "dy/ds=f_free(y)/nu(y)",
            "clock": "dt/ds=1/nu(y)",
            "implementation_note": (
                "endpoint proposals use physical-time spans while accepted "
                "Hermite segments retain both arclength geometry and clock"
            ),
        },
        "initialization": {
            "trajectory": (
                "hash-validated accepted 0--16 ms parent trajectory"
            ),
            "current_state": "accepted parent terminal primitive state",
            "current_coordinate": "accepted parent terminal 470-coordinate state",
            "poincare_memory": (
                "retain start coordinate, section normal, departure=true, "
                "negative_side=false, and the committed switch count"
            ),
            "current_rate": (
                "evaluate exactly once before the first endpoint proposal"
            ),
            "previous_rate": "committed exact 15.75 ms anchor rate",
        },
        "endpoint_proposal": _variable_step_ab2_formula(),
        "segment_validation": {
            "sequence": [
                "propose endpoint coordinate with variable-step AB2",
                "retract endpoint exactly and run every physical audit",
                "evaluate exact original free field at the endpoint",
                "form the cubic Hermite segment from exact endpoint states and rates",
                "bind endpoint trapezoidal integral closure",
                "on every fourth tentative acceptance, exact-retract and exact-evaluate the Hermite midpoint",
                "accept and propagate only after every applicable gate passes",
            ],
            "endpoint_integral_defect": (
                "||(y_1-y_0)/h-(f_0+f_1)/2||_2 / "
                "max(||(y_1-y_0)/h||_2,||(f_0+f_1)/2||_2,tiny)"
            ),
            "endpoint_integral_defect_maximum": (
                MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
            ),
            "blind_midpoint_rate_defect": (
                "||H'(1/2)-f_free(H(1/2))||_2/"
                "max(||f_free(H(1/2))||_2,tiny)"
            ),
            "blind_midpoint_rate_defect_maximum": (
                MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
            ),
            "blind_midpoint_every_nth_tentative_acceptance": (
                BLIND_MIDPOINT_FREQUENCY
            ),
            "blind_midpoint_is_never_used_to_fit_or_correct": True,
            "endpoint_exact_before_propagation": True,
            "failed_candidate_is_never_propagated": True,
        },
        "step_policy": {
            "initial_segment_seconds": INITIAL_SEGMENT_SECONDS,
            "minimum_segment_seconds": MINIMUM_SEGMENT_SECONDS,
            "maximum_segment_seconds": MAXIMUM_SEGMENT_SECONDS,
            "maximum_growth_factor": GROWTH_FACTOR_MAXIMUM,
            "accepted_segments_before_growth": ACCEPTED_SEGMENTS_BEFORE_GROWTH,
            "growth_requires_most_recent_blind_midpoint_pass": True,
            "halve_after_numerical_or_physical_candidate_failure": True,
            "retry_from_last_accepted_endpoint": True,
            "failure_at_minimum_span_is_binding": True,
        },
        "physical_gates": {
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_height_ratio": 0.5,
            "minimum_scattering_optical_depth": 1.0,
            "exact_chart_coordinate_residual_maximum": 5.0e-10,
            "exact_chart_gauge_residual_maximum": 1.0e-10,
            "all_original_storage_and_reaction_free_ledgers": "binding",
            "accepted_history_only": True,
        },
        "terminal_events": {
            "cycle": (
                "after the inherited positive departure, first visit the "
                "negative side and then localize a same-positive-orientation "
                "return to the original Poincare section with an exact event audit"
            ),
            "cycle_hidden_return_defect_maximum": 5.0e-2,
            "cycle_event_requires_exact_free_field_orientation_audit": True,
            "equilibrium_candidate": (
                "speed <=0.1 of the initial committed speed; this stops for a "
                "separate stationary-residual and stability certificate"
            ),
            "budget": "separate inconclusive classification",
        },
        "budgets": {
            "maximum_accepted_segments": MAXIMUM_ACCEPTED_SEGMENTS,
            "maximum_exact_free_field_calls_including_rejections": (
                MAXIMUM_EXACT_FREE_FIELD_CALLS
            ),
            "maximum_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
        },
        "outputs": {
            "every_attempt": (
                "predictor, exact endpoint, exact rate, defects, physical audits, "
                "acceptance reasons, cost, and hashes"
            ),
            "every_accepted_segment": (
                "Hermite coefficients, primitive endpoint, coordinate endpoint, "
                "rate endpoint, elapsed time, Poincare state, and cumulative ledgers"
            ),
            "restart": "bitwise arbitrary-segment restart and suffix replay",
        },
        "outcomes": {
            "cycle": (
                "wide_continuation_cycle_observed_local_transport_passed"
            ),
            "equilibrium_candidate": (
                "wide_continuation_equilibrium_candidate_requires_certificate"
            ),
            "budget": (
                "wide_continuation_inconclusive_acquisition_budget_exhausted"
            ),
            "physical": "wide_continuation_original_free_field_physical_gate_failed",
            "validation": "wide_continuation_endpoint_or_blind_validation_failed",
            "restart": "wide_continuation_restart_or_replay_failed",
        },
        "post_cycle_authorization": (
            "definitions-only matched-path refinement and global cycle-map "
            "certificate; no slow closure or reduced evolution yet"
        ),
        "cost": cost,
    }


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    cold = (
        diagnosis.parent.manifest.parent.arclength._source()._post().manifest
        .transition.manifest.cold.manifest
    )
    with cold.CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": "DEFINITIONS_ONLY",
            })
    with cold.CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(cold.CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": True,
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    helper._write_json(cold.CANONICAL_SUMMARY, catalog)


def _freeze() -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("wide continuation manifest already exists")
    parent_lock = _validate_parent(require_clean=True)
    cost = _cost_projection()
    predictor = _predictor_backtest()
    if not cost["cost_gate_passed"] or not predictor["passed"]:
        raise RuntimeError("wide continuation prospective gate failed")
    contract = _execution_contract(cost)
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", parent_lock)
    helper._write_json(
        CANONICAL_DIRECTORY / "continuation_execution_contract.json", contract
    )
    helper._write_json(CANONICAL_DIRECTORY / "cost_projection.json", cost)
    helper._write_json(
        CANONICAL_DIRECTORY / "endpoint_predictor_backtest.json", predictor
    )
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": {
            str(path.relative_to(ROOT)): helper._sha(path)
            for path in _source_paths()
        },
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "autonomous_truth_preserved": True,
        "endpoint_acquisition_contract_complete": True,
        "curvature_adaptive_continuation_execution_authorized": True,
        "curvature_adaptive_continuation_executed": False,
        "cycle_observed": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join((
            "# Curvature-adaptive arclength continuation manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "The selected truth system remains the autonomous original reaction-free field `dy/dt=f_free(y)`. The endpoint is proposed by variable-step AB2, exactly retracted and physically audited, and then supplied with an exact endpoint field evaluation. A cubic Hermite segment is accepted only after its endpoint integral closure passes; every fourth tentative segment also receives an exact blind midpoint field check.",
            "",
            f"Offline variable-step AB2 replay prevalidates endpoint proposals through 2 ms: the worst steady 2 ms endpoint defect is `{predictor['records']['steady_2ms']['maximum_relative_endpoint_defect']:.6e}`. Four milliseconds remains an exact-validation-controlled attempt, not a promised operational span; its worst diagnostic predictor defect is `{predictor['records']['steady_4ms']['maximum_relative_endpoint_defect']:.6e}`.",
            "",
            f"The span starts at `{INITIAL_SEGMENT_SECONDS:.3e}` s, may grow by at most `{GROWTH_FACTOR_MAXIMUM:.1f}x` to `{MAXIMUM_SEGMENT_SECONDS:.3e}` s, and halves on any candidate failure without propagating the rejected state. The full budget is `{MAXIMUM_EXACT_FREE_FIELD_CALLS}` exact calls and `{MAXIMUM_EXECUTION_WALL_HOURS:.1f}` wall hours. Using the measured parent cost with reserve projects `{cost['reserved_projected_wall_hours']:.3f}` wall hours.",
            "",
            "A detected return authorizes only a matched-path refinement/global cycle-map manifest. Equilibrium stability, slow closure, cycle averaging, and reduced slow evolution remain separately gated.",
            "",
        )),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    arguments = parser.parse_args()
    if not arguments.freeze:
        parser.error("--freeze is required")
    print(json.dumps(_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
