#!/usr/bin/env python3
"""Freeze one accepted-history recovery segment at the diagnosed chart radius."""

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

import run_causal_inner_metric_chart_local_radius_diagnosis_execution_wp10c9d6c7c3b5c4f25fil as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25fim"
CLASSIFICATION = "adaptive_metric_chart_radius_recovery_manifest_frozen"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fin_adaptive_metric_chart_radius_recovery_execution"
)
ARTIFACT = (
    "causal_inner_adaptive_metric_chart_radius_recovery_manifest_"
    "wp10c9d6c7c3b5c4f25fim"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ADAPTIVE_METRIC_CHART_RADIUS_"
    "RECOVERY_MANIFEST_WP10C9D6C7C3B5C4F25FIM_2026-08-24.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_adaptive_metric_chart_radius_recovery_manifest_"
    "wp10c9d6c7c3b5c4f25fim.py"
)
THIS_TEST = (
    "tests/test_causal_inner_adaptive_metric_chart_radius_recovery_manifest_"
    "wp10c9d6c7c3b5c4f25fim.py"
)

PARENT_ELAPSED_SECONDS = parent.manifest.INITIAL_ELAPSED_SECONDS
SEGMENT_SECONDS = 1.0e-3
ENDPOINT_ELAPSED_SECONDS = PARENT_ELAPSED_SECONDS + SEGMENT_SECONDS
EXPECTED_PRIOR_ACCEPTED_SEGMENTS = parent.manifest.INITIAL_ACCEPTED_SEGMENTS
EXPECTED_NEXT_TENTATIVE_SEGMENT = EXPECTED_PRIOR_ACCEPTED_SEGMENTS + 1
BLIND_MIDPOINT_REQUIRED = EXPECTED_NEXT_TENTATIVE_SEGMENT % 4 == 0
MAXIMUM_EXACT_FREE_FIELD_CALLS = 2
MAXIMUM_RETRACTIONS = 2
MAXIMUM_EXECUTION_WALL_HOURS = 2.0
MAXIMUM_SAVED_TARGET_RELATIVE_DEFECT = 2.0e-12
MAXIMUM_SAVED_STATE_RELATIVE_DEFECT = 1.0e-9
MAXIMUM_ENDPOINT_INTEGRAL_DEFECT = (
    parent.manifest.parent.manifest.MAXIMUM_ENDPOINT_INTEGRAL_DEFECT
)
MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT = (
    parent.manifest.parent.manifest.MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT
)
MAXIMUM_METRIC_JACOBIAN_CONDITION = (
    parent.manifest.MAXIMUM_METRIC_JACOBIAN_CONDITION
)
MAXIMUM_METRIC_AUGMENTED_CONDITION = (
    parent.manifest.MAXIMUM_METRIC_AUGMENTED_CONDITION
)
MAXIMUM_PATCH_TRANSITION_CONDITION = (
    parent.manifest.MAXIMUM_PATCH_TRANSITION_CONDITION
)
MAXIMUM_TRANSFORM_INVERSE_CLOSURE = (
    parent.manifest.parent.manifest.MAXIMUM_TRANSFORM_INVERSE_CLOSURE
)
MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT = (
    parent.manifest.parent.manifest.MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT
)
ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE = (
    parent.manifest.ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE
)
METRIC_COORDINATE_RESIDUAL_TOLERANCE = (
    parent.manifest.METRIC_COORDINATE_RESIDUAL_TOLERANCE
)
GAUGE_RESIDUAL_TOLERANCE = parent.manifest.GAUGE_RESIDUAL_TOLERANCE


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
    metrics = helper._read(parent.CANONICAL_DIRECTORY / "diagnosis_metrics.json")
    values = metrics["gate_values"]
    if (
        summary["classification"] != parent.PASS_CLASSIFICATION
        or not summary["passed"]
        or summary["selected_local_radius_seconds"] != SEGMENT_SECONDS
        or summary["new_trajectory"]
        or summary["authorized_next"] != parent.AUTHORIZED_NEXT
        or metrics["classification"] != parent.PASS_CLASSIFICATION
        or not metrics["passed"]
        or values["selected_local_radius_seconds"] != SEGMENT_SECONDS
        or values["strict_retraction_passed"] != [False, True, True]
        or values["physical_passed"] != [True, True, True]
        or values["fresh_patch_passed"] != [None, True, True]
        or values["exact_free_field_calls"] != 0
        or values["new_trajectory"]
        or values["accepted_history_mutated"]
    ):
        raise RuntimeError("local-radius diagnosis evidence changed")
    if require_clean and helper._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("radius-recovery manifest requires a clean tracked tree")
    return {"hashes": hashes, "classification": summary["classification"]}


def _seed() -> dict[str, np.ndarray]:
    helper = _helper()
    diagnosis = helper._load_npz(
        parent.CANONICAL_DIRECTORY / "diagnosis_arrays.npz"
    )
    wide = helper._load_npz(
        parent.manifest.parent.CANONICAL_DIRECTORY / "resume_execution_arrays.npz"
    )
    candidate = parent.manifest.parent.execution._variable_step_ab2(
        wide["current_coordinate470"],
        wide["current_coordinate_rate470_per_s"],
        wide["previous_coordinate_rate470_per_s"],
        SEGMENT_SECONDS,
        wide["previous_span_seconds"],
    )
    selected_target = np.asarray(diagnosis["selected_target_original_coordinate470"])
    if _relative(candidate, selected_target) > MAXIMUM_SAVED_TARGET_RELATIVE_DEFECT:
        raise RuntimeError("diagnosed 1 ms target changed")
    names = (
        "previous_coordinate470",
        "current_coordinate470",
        "previous_primitive_state",
        "current_primitive_state",
        "previous_coordinate_rate470_per_s",
        "current_coordinate_rate470_per_s",
        "previous_span_seconds",
        "elapsed_seconds",
        "accepted_segments_total",
        "current_metric_transform470x470",
        "current_metric_augmented560x560",
        "current_gauge_basis560x90",
        "section_normal470",
        "start_coordinate470",
    )
    seed = {name: np.asarray(wide[name]) for name in names}
    seed.update(
        {
            "segment_seconds": np.asarray(SEGMENT_SECONDS),
            "candidate_target470": candidate,
            "diagnosed_target470": selected_target,
            "diagnosed_primitive_state": np.asarray(
                diagnosis["selected_primitive_state"]
            ),
            "diagnosed_metric_transform470x470": np.asarray(
                diagnosis["selected_metric_transform470x470"]
            ),
            "diagnosed_metric_augmented560x560": np.asarray(
                diagnosis["selected_metric_augmented560x560"]
            ),
            "diagnosed_gauge_basis560x90": np.asarray(
                diagnosis["selected_gauge_basis560x90"]
            ),
        }
    )
    if (
        float(seed["elapsed_seconds"]) != PARENT_ELAPSED_SECONDS
        or int(seed["accepted_segments_total"])
        != EXPECTED_PRIOR_ACCEPTED_SEGMENTS
        or float(seed["previous_span_seconds"]) != 2.0e-3
        or not BLIND_MIDPOINT_REQUIRED
    ):
        raise RuntimeError("radius-recovery accepted history changed")
    return seed


def _contract() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "authorized_execution": AUTHORIZED_NEXT,
        "truth_system": {
            "field": "autonomous original reaction-free field dy/dt=f_free(y)",
            "state": "original primitive state u",
            "physical_coordinate": "original q=C(u)",
            "metric_chart": "numerical preconditioner only",
            "strict_retraction_status": "physical AND closure AND chart condition",
            "fixed_Q_rate_or_reaction": "forbidden",
        },
        "history": {
            "parent_elapsed_seconds": PARENT_ELAPSED_SECONDS,
            "previous_span_seconds": 2.0e-3,
            "recovery_segment_seconds": SEGMENT_SECONDS,
            "predictor": "variable-step AB2 in original q",
            "tentative_segment_number": EXPECTED_NEXT_TENTATIVE_SEGMENT,
            "blind_midpoint_required": BLIND_MIDPOINT_REQUIRED,
            "diagnosed_endpoint_retraction_replayed": True,
            "failed_2_ms_candidate_never_propagated": True,
        },
        "scope": {
            "new_exact_free_field_calls": MAXIMUM_EXACT_FREE_FIELD_CALLS,
            "new_retractions": MAXIMUM_RETRACTIONS,
            "new_accepted_segments_maximum": 1,
            "new_physical_time_seconds_maximum": SEGMENT_SECONDS,
            "maximum_wall_hours": MAXIMUM_EXECUTION_WALL_HOURS,
            "fixed_Q_calls": 0,
            "reaction_calls": 0,
            "nonlinear_roots": 0,
            "BDF_microsteps": 0,
        },
        "gates": {
            "original_coordinate_residual_tolerance": ORIGINAL_COORDINATE_RESIDUAL_TOLERANCE,
            "metric_coordinate_residual_tolerance": METRIC_COORDINATE_RESIDUAL_TOLERANCE,
            "gauge_residual_tolerance": GAUGE_RESIDUAL_TOLERANCE,
            "maximum_metric_jacobian_condition": MAXIMUM_METRIC_JACOBIAN_CONDITION,
            "maximum_metric_augmented_condition": MAXIMUM_METRIC_AUGMENTED_CONDITION,
            "maximum_patch_transition_condition": MAXIMUM_PATCH_TRANSITION_CONDITION,
            "maximum_transform_inverse_closure": MAXIMUM_TRANSFORM_INVERSE_CLOSURE,
            "maximum_coordinate_reconstruction_defect": MAXIMUM_COORDINATE_RECONSTRUCTION_DEFECT,
            "maximum_saved_state_relative_defect": MAXIMUM_SAVED_STATE_RELATIVE_DEFECT,
            "maximum_endpoint_integral_defect": MAXIMUM_ENDPOINT_INTEGRAL_DEFECT,
            "maximum_blind_midpoint_rate_defect": MAXIMUM_BLIND_MIDPOINT_RATE_DEFECT,
            "minimum_reconstruction_factor": 1.0 - 1.0e-12,
            "maximum_height_ratio": 0.5,
            "minimum_scattering_optical_depth": 1.0,
            "all_reaction_free_ledgers": True,
            "checkpoint_roundtrip_bitwise": True,
            "history_replay_bitwise": True,
        },
        "decision": {
            "pass": "adaptive metric chart radius recovery passed",
            "pass_authorizes": "definitions-only adaptive continuation manifest",
            "physical_failure": "original physical gate failed",
            "numerical_failure": "retraction, field, defect, or restart gate failed",
        },
        "forbidden": [
            "retroactively accept the rejected 2 ms candidate",
            "propagate before the blind midpoint passes",
            "alter the original physical field or ledgers",
            "relax any gate",
            "authorize complete-cycle or reduced slow execution directly",
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
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": helper._sha(path),
                    "scientific_status": "SUPPORTED",
                }
            )
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
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    helper._write_json(summary_path, catalog)


def _freeze() -> dict:
    helper = _helper()
    lock = _validate_parent(require_clean=True)
    seed = _seed()
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("radius-recovery manifest already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "recovery_contract.json", _contract())
    _save_npz(CANONICAL_DIRECTORY / "recovery_seed.npz", seed)
    helper._write_json(CANONICAL_DIRECTORY / "input_lock.json", lock)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": True,
        "definitions_only": True,
        "adaptive_metric_chart_radius_recovery_authorized": True,
        "adaptive_metric_chart_radius_recovery_executed": False,
        "new_trajectory": False,
        "cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "definition_commit": helper._git("rev-parse", "HEAD"),
            "definition_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                THIS_RUNNER: helper._sha(ROOT / THIS_RUNNER),
                THIS_TEST: helper._sha(ROOT / THIS_TEST),
                parent.THIS_RUNNER: helper._sha(ROOT / parent.THIS_RUNNER),
                parent.THIS_TEST: helper._sha(ROOT / parent.THIS_TEST),
                parent.manifest.STRICT_ATLAS_SOURCE: helper._sha(
                    ROOT / parent.manifest.STRICT_ATLAS_SOURCE
                ),
                parent.manifest.parent.source.THIS_RUNNER: helper._sha(
                    ROOT / parent.manifest.parent.source.THIS_RUNNER
                ),
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
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
        "\n".join(
            (
                "# Adaptive metric-chart radius recovery manifest",
                "",
                f"Classification: `{CLASSIFICATION}`.",
                "",
                "This definitions-only package replays the diagnosed 1.00 ms strict endpoint retraction, evaluates the original slow field, and requires the tentative-segment-92 blind midpoint before accepting any new history.",
                "",
                "The rejected 2.00 ms candidate remains rejected and is never propagated.",
                "",
                f"Authorized next artifact: `{AUTHORIZED_NEXT}`.",
                "",
            )
        ),
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
