#!/usr/bin/env python3
"""Freeze a conservative metric-chart atlas overlap preflight."""

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

import run_causal_inner_coordinate_chart_conditioning_diagnosis_wp10c9d6c7c3b5c4f25fib as diagnosis  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fic"
CLASSIFICATION = "conservative_metric_chart_atlas_overlap_manifest_frozen"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25fid_metric_chart_atlas_overlap_preflight"
ARTIFACT = (
    "causal_inner_conservative_metric_chart_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25fic"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_CONSERVATIVE_METRIC_CHART_ATLAS_"
    "MANIFEST_WP10C9D6C7C3B5C4F25FIC_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_conservative_metric_chart_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25fic.py"
)
THIS_TEST = (
    "tests/test_causal_inner_conservative_metric_chart_atlas_manifest_"
    "wp10c9d6c7c3b5c4f25fic.py"
)

ANCHOR_ATTEMPT = 82
OVERLAP_ATTEMPT = 83
RAW_WARNING_CONDITION = 2.0e3
RAW_HISTORICAL_HARD_CONDITION = 2.5e3
MAXIMUM_METRIC_JACOBIAN_CONDITION = 10.0
MAXIMUM_METRIC_AUGMENTED_CONDITION = 10.0
MAXIMUM_TRANSFORM_INVERSE_CLOSURE = 1.0e-10
MAXIMUM_ORIGINAL_COORDINATE_ROUNDTRIP_DEFECT = 1.0e-10
MAXIMUM_RATE_PUSH_PULL_DEFECT = 1.0e-10
MAXIMUM_SAVED_STATE_RELATIVE_DEFECT = 1.0e-9
ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE = 1.0e-10
METRIC_COORDINATE_RESIDUAL_TOLERANCE = 1.0e-9
GAUGE_RESIDUAL_TOLERANCE = 1.0e-10
MAXIMUM_PATCH_TRANSITION_CONDITION = 10.0
MAXIMUM_RETRACTIONS = 3
MAXIMUM_EXECUTION_WALL_HOURS = 1.0


def _helper():
    return diagnosis._helper()


def _validate_diagnosis(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(diagnosis.CANONICAL_DIRECTORY)
    summary = helper._read(diagnosis.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        diagnosis.CANONICAL_DIRECTORY / "conditioning_metrics.json"
    )
    values = metrics["gate_values"]
    if (
        summary["classification"] != diagnosis.METRIC_CLASSIFICATION
        or not summary["passed"]
        or not summary["atlas_supported"]
        or summary["authorized_next"] != diagnosis.AUTHORIZED_METRIC_NEXT
        or metrics["classification"] != diagnosis.METRIC_CLASSIFICATION
        or not metrics["passed"]
        or not metrics["atlas_supported"]
        or not values["method_passed"]
        or not values["rank_passed"]
        or not values["row_equilibrated_gate_passed"]
        or not values["block_whitened_gate_passed"]
        or values["new_exact_free_field_calls"] != 0
        or values["new_retractions"] != 0
        or values["new_trajectory_segments"] != 0
    ):
        raise RuntimeError("metric-chart diagnosis changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("metric-chart manifest requires a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "metrics": metrics}


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "authorized_execution": AUTHORIZED_NEXT,
        "mathematical_chart": {
            "truth_state": "original primitive state u in R^560",
            "original_coordinate": "q=C(u) in R^470 remains binding for physics",
            "metric_coordinate": "z_k=W_k(q-q_k)",
            "metric_rate": "dz_k/dt=W_k DC(u) du/dt",
            "transform": (
                "block diagonal inverse-square-root row-Gram whitening of the "
                "162 physical, 280 memory, and 28 departure coordinate blocks"
            ),
            "transition": (
                "carry the primitive state, evaluate original q exactly, and "
                "encode it in the next accepted anchor chart"
            ),
            "conservation": (
                "all ledgers, guards, Poincare events, stored observables, and "
                "physical acceptance remain in original primitive/original q space"
            ),
        },
        "anchors": {
            "accepted_anchor_attempt": ANCHOR_ATTEMPT,
            "overlap_candidate_attempt": OVERLAP_ATTEMPT,
            "raw_warning_condition": RAW_WARNING_CONDITION,
            "historical_raw_hard_condition_preserved": (
                RAW_HISTORICAL_HARD_CONDITION
            ),
            "switch_policy": (
                "construct a new metric patch at the last accepted state once "
                "the raw diagnostic condition reaches the warning threshold"
            ),
        },
        "scope": {
            "new_exact_coordinate_jacobians": 2,
            "new_exact_free_field_calls": 0,
            "new_retractions": MAXIMUM_RETRACTIONS,
            "new_trajectory_segments": 0,
            "new_physical_time_seconds": 0.0,
            "fixed_Q_calls": 0,
            "reaction_calls": 0,
            "nonlinear_roots": 0,
            "BDF_microsteps": 0,
            "maximum_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
        },
        "required_retractions": [
            "anchor82_to_saved_overlap83",
            "bitwise_replay_anchor82_to_saved_overlap83",
            "reanchored_overlap83_to_anchor82",
        ],
        "gates": {
            "maximum_metric_jacobian_condition": (
                MAXIMUM_METRIC_JACOBIAN_CONDITION
            ),
            "maximum_metric_augmented_condition": (
                MAXIMUM_METRIC_AUGMENTED_CONDITION
            ),
            "maximum_transform_inverse_closure": (
                MAXIMUM_TRANSFORM_INVERSE_CLOSURE
            ),
            "maximum_original_coordinate_roundtrip_defect": (
                MAXIMUM_ORIGINAL_COORDINATE_ROUNDTRIP_DEFECT
            ),
            "maximum_rate_push_pull_defect": MAXIMUM_RATE_PUSH_PULL_DEFECT,
            "maximum_saved_state_relative_defect": (
                MAXIMUM_SAVED_STATE_RELATIVE_DEFECT
            ),
            "original_coordinate_residual_tolerance": (
                ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE
            ),
            "metric_coordinate_residual_tolerance": (
                METRIC_COORDINATE_RESIDUAL_TOLERANCE
            ),
            "gauge_residual_tolerance": GAUGE_RESIDUAL_TOLERANCE,
            "maximum_patch_transition_condition": (
                MAXIMUM_PATCH_TRANSITION_CONDITION
            ),
            "forward_replay_bitwise": True,
            "all_original_physical_gates_except_historical_raw_condition": True,
        },
        "decision": {
            "all_gates_pass": (
                "metric_chart_atlas_overlap_passed_boundary_crossing_manifest_authorized"
            ),
            "metric_or_transition_gate_fails": (
                "metric_chart_atlas_failed_no_boundary_crossing_authorized"
            ),
            "original_physical_gate_fails": (
                "original_free_field_physical_failure_no_continuation_authorized"
            ),
        },
        "forbidden": [
            "modify the original coordinate definition",
            "evaluate physics in metric coordinates",
            "relax the historical raw gate retroactively",
            "advance physical time",
            "authorize a cycle or reduced slow evolution",
        ],
    }


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
                "scientific_status": "DEFINITIONS_ONLY",
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
        "passed": summary["passed"],
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


def _canonicalize(lock: dict) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("metric-chart atlas manifest already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "atlas_contract.json", _contract())
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", {
        "diagnosis_hashes": lock["hashes"],
        "diagnosis_classification": lock["summary"]["classification"],
        "diagnosis_gate_values": lock["metrics"]["gate_values"],
    })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "metric_chart_atlas_preflight_authorized": True,
        "metric_chart_atlas_executed": False,
        "trajectory_authorized": False,
        "cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "implementation_commit": helper._git("rev-parse", "HEAD"),
        "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": {
            THIS_RUNNER: helper._sha(ROOT / THIS_RUNNER),
            THIS_TEST: helper._sha(ROOT / THIS_TEST),
            diagnosis.THIS_RUNNER: helper._sha(ROOT / diagnosis.THIS_RUNNER),
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
            "# Conservative metric-chart atlas manifest",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "The original primitive state, original 470 coordinates, autonomous free field, physical guards, ledgers, and Poincare observables remain binding. A block-whitened local coordinate is introduced only as a numerical metric for retraction and chart admissibility.",
            "",
            "The preflight is restricted to the saved accepted attempt 82 and rejected overlap attempt 83. It may assemble two coordinate Jacobians and perform three saved-target retractions, but it may not evaluate a new field or advance physical time.",
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
    lock = _validate_diagnosis(require_clean=True)
    summary = _canonicalize(lock)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
