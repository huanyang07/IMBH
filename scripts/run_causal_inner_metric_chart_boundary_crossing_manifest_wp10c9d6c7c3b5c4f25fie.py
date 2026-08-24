#!/usr/bin/env python3
"""Freeze one conservative metric-chart boundary-crossing segment."""

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

import run_causal_inner_metric_chart_atlas_overlap_preflight_wp10c9d6c7c3b5c4f25fid as parent  # noqa: E402


execution = parent.parent
SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fie"
CLASSIFICATION = "metric_chart_boundary_crossing_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fif_metric_chart_boundary_crossing_execution"
)
ARTIFACT = (
    "causal_inner_metric_chart_boundary_crossing_manifest_"
    "wp10c9d6c7c3b5c4f25fie"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_METRIC_CHART_BOUNDARY_CROSSING_"
    "MANIFEST_WP10C9D6C7C3B5C4F25FIE_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_metric_chart_boundary_crossing_manifest_"
    "wp10c9d6c7c3b5c4f25fie.py"
)
THIS_TEST = (
    "tests/test_causal_inner_metric_chart_boundary_crossing_manifest_"
    "wp10c9d6c7c3b5c4f25fie.py"
)

PARENT_ELAPSED_SECONDS = 0.11125000000000008
SEGMENT_SECONDS = 2.5e-4
ENDPOINT_ELAPSED_SECONDS = PARENT_ELAPSED_SECONDS + SEGMENT_SECONDS
EXPECTED_PRIOR_ACCEPTED_SEGMENTS = 71
EXPECTED_NEXT_TENTATIVE_SEGMENT = 72
BLIND_MIDPOINT_REQUIRED = True
HISTORICAL_RAW_CONDITION = 2.5e3
MAXIMUM_METRIC_JACOBIAN_CONDITION = 10.0
MAXIMUM_METRIC_AUGMENTED_CONDITION = 10.0
MAXIMUM_PATCH_TRANSITION_CONDITION = 10.0
MAXIMUM_TRANSFORM_INVERSE_CLOSURE = 1.0e-10
MAXIMUM_SAVED_TARGET_RELATIVE_DEFECT = 2.0e-12
MAXIMUM_SAVED_STATE_RELATIVE_DEFECT = 1.0e-9
MAXIMUM_SAVED_RATE_RELATIVE_DEFECT = 1.0e-9
MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT = 5.0e-12
MAXIMUM_ENDPOINT_INTEGRAL_DEFECT = 2.0e-2
MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT = 2.0e-2
ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE = 1.0e-10
METRIC_COORDINATE_RESIDUAL_TOLERANCE = 1.0e-9
GAUGE_RESIDUAL_TOLERANCE = 1.0e-10
MAXIMUM_EXACT_FREE_FIELD_CALLS = 2
MAXIMUM_INDEPENDENT_METRIC_JACOBIAN_AUDITS = 2
MAXIMUM_RETRACTIONS = 3
MAXIMUM_EXECUTION_WALL_HOURS = 2.0


def _helper():
    return parent._helper()


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(a - b)
        / max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), np.finfo(float).tiny)
    )


def _validate_parent(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = helper._read(parent.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(parent.CANONICAL_DIRECTORY / "overlap_metrics.json")
    values = metrics["gate_values"]
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or not summary["metric_chart_atlas_overlap_passed"]
        or not summary["original_physics_preserved"]
        or summary["new_trajectory"]
        or not summary["boundary_crossing_manifest_authorized"]
        or summary["authorized_next"] != parent.AUTHORIZED_NEXT
        or metrics["classification"] != parent.PASS_CLASSIFICATION
        or not metrics["passed"]
        or not values["all_retractions_passed"]
        or not values["forward_replay_bitwise"]
        or values["new_exact_free_field_calls"] != 0
        or values["new_trajectory_segments"] != 0
        or values["maximum_metric_augmented_condition"]
        > parent.manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
    ):
        raise RuntimeError("metric-chart overlap authorization changed")
    execution_hashes = helper._validate_checksums(execution.CANONICAL_DIRECTORY)
    execution_summary = helper._read(execution.CANONICAL_DIRECTORY / "summary.json")
    execution_metrics = helper._read(
        execution.CANONICAL_DIRECTORY / "continuation_execution_metrics.json"
    )
    execution_values = execution_metrics["gate_values"]
    if (
        execution_summary["classification"] != execution.PHYSICAL_CLASSIFICATION
        or execution_summary["passed"]
        or execution_values["accepted_segments"]
        != EXPECTED_PRIOR_ACCEPTED_SEGMENTS
        or execution_values["terminal_elapsed_seconds"]
        != PARENT_ELAPSED_SECONDS
        or not execution_values["restart_roundtrip_bitwise"]
        or not execution_values["suffix_replay_bitwise"]
        or execution_metrics["fixed_Q_physical_rate_calls"] != 0
        or execution_metrics["fixed_Q_reaction_calls"] != 0
    ):
        raise RuntimeError("terminal continuation evidence changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("boundary-crossing manifest requires a clean tracked tree")
    return {
        "overlap_hashes": hashes,
        "overlap_classification": summary["classification"],
        "continuation_hashes": execution_hashes,
        "continuation_classification": execution_summary["classification"],
    }


def _seed() -> dict[str, np.ndarray]:
    helper = _helper()
    history_path = (
        execution.CANONICAL_DIRECTORY / "continuation_execution_arrays.npz"
    )
    numeric_history_names = (
        "trajectory_coordinates",
        "trajectory_primitive_states",
        "accepted_endpoint_rates470_per_s",
        "accepted_segment_seconds",
        "terminal_coordinate_rate470_per_s",
        "section_normal470",
        "start_coordinate470",
    )
    with np.load(history_path, allow_pickle=False) as payload:
        history = {
            name: np.asarray(payload[name]) for name in numeric_history_names
        }
    overlap = helper._load_npz(parent.CANONICAL_DIRECTORY / "overlap_arrays.npz")
    witness = parent._selected_witnesses()
    trajectory = np.asarray(history["trajectory_coordinates"], dtype=float)
    states = np.asarray(history["trajectory_primitive_states"], dtype=float)
    rates = np.asarray(history["accepted_endpoint_rates470_per_s"], dtype=float)
    spans = np.asarray(history["accepted_segment_seconds"], dtype=float)
    if (
        trajectory.shape != (136, 470)
        or states.shape != (136, 112, 5)
        or rates.shape != (EXPECTED_PRIOR_ACCEPTED_SEGMENTS, 470)
        or spans.shape != (EXPECTED_PRIOR_ACCEPTED_SEGMENTS,)
        or spans[-1] != SEGMENT_SECONDS
    ):
        raise RuntimeError("terminal continuation history shape changed")
    np.testing.assert_array_equal(
        trajectory[-1], overlap["anchor_original_coordinate470"]
    )
    np.testing.assert_array_equal(states[-1], overlap["anchor_primitive_state"])
    np.testing.assert_array_equal(rates[-1], history["terminal_coordinate_rate470_per_s"])
    candidate = execution._variable_step_ab2(
        trajectory[-1], rates[-1], rates[-2], SEGMENT_SECONDS, spans[-1]
    )
    saved_target = np.asarray(overlap["overlap_original_coordinate470"], dtype=float)
    saved_state = np.asarray(overlap["saved_overlap_primitive_state"], dtype=float)
    saved_rate = np.asarray(witness["coordinate_free_rates470_per_s"][1], dtype=float)
    target_defect = _relative(candidate, saved_target)
    if target_defect > MAXIMUM_SAVED_TARGET_RELATIVE_DEFECT:
        raise RuntimeError(
            "prospective AB2 target no longer matches saved recovered boundary"
        )
    return {
        "previous_coordinate470": trajectory[-2],
        "current_coordinate470": trajectory[-1],
        "previous_primitive_state": states[-2],
        "current_primitive_state": states[-1],
        "previous_coordinate_rate470_per_s": rates[-2],
        "current_coordinate_rate470_per_s": rates[-1],
        "previous_span_seconds": np.asarray(spans[-1]),
        "segment_seconds": np.asarray(SEGMENT_SECONDS),
        "candidate_target470": candidate,
        "saved_boundary_target470": saved_target,
        "saved_boundary_primitive_state": saved_state,
        "saved_boundary_coordinate_rate470_per_s": saved_rate,
        "section_normal470": np.asarray(history["section_normal470"]),
        "start_coordinate470": np.asarray(history["start_coordinate470"]),
        "anchor_metric_transform470x470": np.asarray(
            overlap["anchor_metric_transform470x470"]
        ),
        "anchor_metric_augmented560x560": np.asarray(
            overlap["anchor_metric_augmented560x560"]
        ),
        "anchor_gauge_basis560x90": np.asarray(
            overlap["anchor_gauge_basis560x90"]
        ),
    }


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "authorized_execution": AUTHORIZED_NEXT,
        "truth_system": {
            "field": "autonomous original reaction-free field dy/dt=f_free(y)",
            "state": "original primitive state u",
            "physical_coordinate": "original q=C(u)",
            "metric_coordinate": "local z=W(q-q_anchor), numerical only",
            "fixed_Q_rate_or_reaction": "forbidden",
            "external_clock_or_phase": "forbidden",
        },
        "history": {
            "parent_elapsed_seconds": PARENT_ELAPSED_SECONDS,
            "previous_and_current_states": "hash-locked accepted f25fi history",
            "previous_and_current_rates": "hash-locked exact f25fi fields",
            "previous_span_seconds": SEGMENT_SECONDS,
            "new_segment_seconds": SEGMENT_SECONDS,
            "predictor": "unchanged equal-step AB2 in original q",
            "tentative_segment_number": EXPECTED_NEXT_TENTATIVE_SEGMENT,
            "blind_midpoint_required": BLIND_MIDPOINT_REQUIRED,
        },
        "chart_policy": {
            "retraction_patch": "accepted attempt-82 metric patch",
            "endpoint_acceptance_patch": "new exact endpoint metric patch",
            "historical_raw_condition": HISTORICAL_RAW_CONDITION,
            "historical_raw_condition_role": "reported diagnostic, not binding",
            "rank_470_remains_binding": True,
            "all_original_physical_gates_except_raw_condition_remain_binding": True,
            "all_ledgers_remain_in_original_coordinates": True,
        },
        "scope": {
            "new_exact_free_field_calls": MAXIMUM_EXACT_FREE_FIELD_CALLS,
            "new_independent_metric_jacobian_audits": (
                MAXIMUM_INDEPENDENT_METRIC_JACOBIAN_AUDITS
            ),
            "new_retractions": MAXIMUM_RETRACTIONS,
            "new_accepted_segments_maximum": 1,
            "new_physical_time_seconds_maximum": SEGMENT_SECONDS,
            "fixed_Q_calls": 0,
            "reaction_calls": 0,
            "nonlinear_roots": 0,
            "BDF_microsteps": 0,
            "maximum_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
        },
        "gates": {
            "original_coordinate_residual_tolerance": (
                ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE
            ),
            "metric_coordinate_residual_tolerance": (
                METRIC_COORDINATE_RESIDUAL_TOLERANCE
            ),
            "gauge_residual_tolerance": GAUGE_RESIDUAL_TOLERANCE,
            "maximum_metric_jacobian_condition": (
                MAXIMUM_METRIC_JACOBIAN_CONDITION
            ),
            "maximum_metric_augmented_condition": (
                MAXIMUM_METRIC_AUGMENTED_CONDITION
            ),
            "maximum_patch_transition_condition": (
                MAXIMUM_PATCH_TRANSITION_CONDITION
            ),
            "maximum_transform_inverse_closure": (
                MAXIMUM_TRANSFORM_INVERSE_CLOSURE
            ),
            "maximum_saved_target_relative_defect": (
                MAXIMUM_SAVED_TARGET_RELATIVE_DEFECT
            ),
            "maximum_saved_state_relative_defect": (
                MAXIMUM_SAVED_STATE_RELATIVE_DEFECT
            ),
            "maximum_saved_rate_relative_defect": (
                MAXIMUM_SAVED_RATE_RELATIVE_DEFECT
            ),
            "maximum_coordinate_reconstruction_defect": (
                MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT
            ),
            "maximum_endpoint_integral_defect": MAXIMUM_ENDPOINT_INTEGRAL_DEFECT,
            "maximum_blind_midpoint_rate_defect": (
                MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
            ),
            "endpoint_retraction_replay_bitwise": True,
            "checkpoint_roundtrip_bitwise": True,
        },
        "decision": {
            "pass": (
                "metric_chart_boundary_crossing_passed_short_suffix_manifest_authorized"
            ),
            "physical_failure": (
                "metric_chart_boundary_crossing_original_physical_gate_failed"
            ),
            "numerical_failure": "metric_chart_boundary_crossing_numerical_gate_failed",
        },
        "forbidden": [
            "retroactively pass the rejected f25fi attempt 83",
            "modify the original coordinate or physical field",
            "relax rank, reconstruction, height, optical-depth, or ledger gates",
            "propagate a rejected endpoint",
            "authorize a cycle or reduced slow evolution",
        ],
    }


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = ROOT / "results/manifests/canonical_artifacts.csv"
    summary_path = ROOT / "results/manifests/canonical_summary.json"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": helper._sha(path),
                "scientific_status": "SUPPORTED",
            })
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(summary_path)
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
    helper._write_json(summary_path, catalog)


def _freeze() -> dict:
    helper = _helper()
    lock = _validate_parent(require_clean=True)
    seed = _seed()
    contract = _contract()
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("boundary-crossing manifest already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "boundary_contract.json", contract)
    _save_npz(CANONICAL_DIRECTORY / "boundary_seed.npz", seed)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", lock)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "metric_chart_boundary_crossing_authorized": True,
        "metric_chart_boundary_crossing_executed": False,
        "new_trajectory": False,
        "cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "definition_commit": helper._git("rev-parse", "HEAD"),
        "definition_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": {
            THIS_RUNNER: helper._sha(ROOT / THIS_RUNNER),
            THIS_TEST: helper._sha(ROOT / THIS_TEST),
            parent.THIS_RUNNER: helper._sha(ROOT / parent.THIS_RUNNER),
            parent.ATLAS_SOURCE: helper._sha(ROOT / parent.ATLAS_SOURCE),
            execution.THIS_RUNNER: helper._sha(ROOT / execution.THIS_RUNNER),
            execution.source.THIS_RUNNER: helper._sha(
                ROOT / execution.source.THIS_RUNNER
            ),
        },
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
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
            "# Metric-chart boundary-crossing manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "This definitions-only package freezes one 0.25 ms AB2 segment from the accepted 111.25 ms endpoint. The endpoint and required blind midpoint use the original autonomous reaction-free field and original physical coordinates; metric charts affect only numerical retraction and conditioning.",
            "",
            "The historical raw condition threshold remains a diagnostic boundary. Rank, metric conditioning, all original physical and ledger gates, endpoint integration, restart, and replay remain binding.",
            "",
            f"Authorized next artifact: `{AUTHORIZED_NEXT}`.",
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
